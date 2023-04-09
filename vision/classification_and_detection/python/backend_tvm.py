"""
TVM backend for MLPerf inference vision benchmark
Developers: Alexander Peskov, Thierry Moreau, Grigori Fursin
"""

import backend

import tvm
from tvm import auto_scheduler
from tvm.contrib import graph_executor
from tvm.runtime import vm as runtime_vm

import numpy as np

import os
import multiprocessing


class WorkerDescriptor:
    """
    Class that provides an API for loading and inferencing a model
    """
    executable: typing.Any
    executable_type: str = "virtual_machine"
    max_batchsize: int

    def init_executable(self,
                        model_path: typing.Any,
                        max_batchsize: int,
                        inputs: typing.Any = None,
                        outputs: typing.Any = None) -> typing.Any:
        """
        Load of the model in the format of the VirtualMachine / GraphExecutor
        """
        self.inputs = inputs
        self.outputs = outputs

        self.max_batchsize = max_batchsize

        if model_path.endswith('.so') or model_path.endswith('.dylib'):
            compiled_model = model_path
            if not os.path.isfile(compiled_model):
                print()
                raise RuntimeError(
                    f"Error: Model file {compiled_model} not found!"
                )
        else:
            raise RuntimeError(
                f"Error: The specified path ({model_path}) does not match path to the compiled model!"
            )

        print('TVM: loading model ' + compiled_model)

        mod = tvm.runtime.load_module(compiled_model)
        work_dir = os.path.dirname(model_path)

        if os.path.isfile(os.path.join(work_dir, "vm_exec_code.ro")):
            self.executable_type = "virtual_machine"

            with open(os.path.join(work_dir, "vm_exec_code.ro"), "rb") as file:
                vm_bytes = file.read()

            vm_exec = tvm.runtime.vm.Executable.load_exec(vm_bytes, mod)

            for sub_dir in next(os.walk(work_dir))[1]:
                if sub_dir.endswith("-tvm-tmp"):
                    path_consts = os.path.join(
                        work_dir, sub_dir + "/consts")
                    break

            vm_exec.mod["load_late_bound_consts"](path_consts)

            self.executable = runtime_vm.VirtualMachine(vm_exec, device)
        else:
            self.executable_type = "graph_executor"
            self.executable = graph_executor.GraphModule(
                mod['default'](device))

        print(f"WorkerData::init_executable[{os.getpid()}]: {executable}")

    def inference(self, data: typing.Any) -> typing.Any:
        """
        Inference with VirtualMachine / GraphExecutor API
        """
        print(f"WorkerData::inference[{os.getpid()}]: {self.executable}")

        batch_size = self.max_batchsize
        for iname, item in data.items():
            batch_size = len(data)
            if batch_size < self.max_batchsize:
                # Fill in with the first tensor
                item_extra = np.stack(
                    [item[0]] * (self.max_batchsize - batch_size))
                item = np.vstack((item, item_extra))
            elif batch_size > self.max_batchsize:
                raise ValueError(
                    "Internal MLPerf error: dynamic batch size > max batch size")

            input_idx = self.inputs.index(iname)
            self.executable.set_input("main", tvm.nd.array(item))

        self.executable.run()

        tvm_output = []
        output_order = range(len(self.vm.get_outputs())
                             ) if not self.output_order else self.output_order
        for i in output_order:
            # Take only the output of batch size for dynamic batches
            tvm_output.append(self.vm.module["get_output"](
                i).asnumpy()[:batch_size])

        return tvm_output


class PoolBackendTVM(backend.Backend):
    """
    Asynchronous launcher based on a pool of workers
    """
    pool: multiprocessing.Pool
    worker_descriptor: WorkerDescriptor = WorkerDescriptor()
    max_batchsize: int = None

    def __init__(self, num_processes: int = multiprocessing.cpu_count()):
        self.num_processes: int = num_processes

    def version(self):
        return tvm.__version__

    def name(self):
        return "tvm"

    @staticmethod
    def worker_initializer(model_path: str, inputs=None, outputs=None) -> None:
        print(
            f"PoolBackendTVM::worker_initializer[{os.getpid()}]: {model_path}")
        PoolBackendTVM.worker_descriptor.init_executable(
            model_path, self.max_batchsize, inputs, outputs)

    def worker_handler(self, feed: typing.Any) -> typing.Any:
        print(
            f"PoolBackendTVM::worker_handler[{os.getpid()}][{self.worker_descriptor.executable}]: {feed}")
        return self.worker_descriptor.inference(feed)

    def load(self, model_path: str, inputs=None, outputs=None) -> PoolBackendTVM:
        print(f"PoolBackendTVM::load[{os.getpid()}]: {model_path}")

        PoolBackendTVM.pool = multiprocessing.Pool(
            self.num_processes,
            initializer=PoolBackendTVM.worker_initializer,
            initargs=(model_path, inputs, outputs)
        )
        return self

    def predict(
            self, feed: typing.Any, async_mode: bool = True
    ) -> typing.Union[typing.Any, multiprocessing.pool.ApplyResult]:
        print(f"PoolBackendTVM::predict[{os.getpid()}]: {feed}")
        if async_mode:
            return self.predict_async(feed)
        else:
            return self.predict_sync(feed)

    def predict_sync(self, feed: typing.Any) -> typing.Any:
        print(f"PoolBackendTVM::predict_sync[{os.getpid()}]: {feed}")
        return self.pool.apply_async(self.worker_handler, args=(feed,)).get()

    def predict_async(self, feed: typing.Any) -> multiprocessing.pool.ApplyResult:
        print(f"PoolBackendTVM::predict_async[{os.getpid()}]: {feed}")
        resp: multiprocessing.pool.ApplyResult = self.pool.apply_async(
            self.worker_handler, args=(feed,))
        return resp

    @staticmethod
    def async_response(async_responses: typing.List[multiprocessing.pool.ApplyResult]) -> typing.List[typing.Any]:
        return [resp.get() for resp in async_responses]

    def finish(self) -> None:
        self.pool.terminate()


class AsyncBackendTVM(backend.Backend):
    """
    Asynchronous launcher based on processes and a concurrent task queue
    """
    concurrent_queue: multiprocessing.Queue = multiprocessing.Queue(
        utils.QUEUE_SIZE)
    manager: multiprocessing.Manager = multiprocessing.Manager()
    response_map: typing.Dict = manager.dict()
    max_batchsize: int = None

    def __init__(self, num_processes: int = multiprocessing.cpu_count()):
        print(f"AsyncBackendTVM::__init__[{os.getpid()}]")
        self.num_processes: int = num_processes
        self.workers: typing.List[multiprocessing.Process] = []

    @staticmethod
    def _async_inference(samples: typing.Any, descriptor: WorkerDescriptor):
        print(f"AsyncBackendTVM::_async_inference[{os.getpid()}]: {samples}")
        return descriptor.inference(samples)

    @staticmethod
    def _async_response(descriptor: WorkerDescriptor) -> bool:
        print(
            f"AsyncBackendTVM::_async_response[{os.getpid()}]: {descriptor.executable}")
        sample_id, samples = AsyncBackendTVM.concurrent_queue.get(block=True)
        if samples == -1:   # End marker
            return False
        inference_response = AsyncBackendTVM._async_inference(
            samples, descriptor)
        AsyncBackendTVM.response_map[sample_id] = inference_response
        return True

    @staticmethod
    def _worker_action(descriptor: WorkerDescriptor):
        print(
            f"AsyncBackendTVM::_worker_action[{os.getpid()}]: {descriptor.executable}")
        while AsyncBackendTVM._async_response(descriptor):
            # Empty body
            pass

    @staticmethod
    def _create_worker(descriptor: WorkerDescriptor) -> multiprocessing.Process:
        print(
            f"AsyncBackendTVM::_create_worker[{os.getpid()}]: {descriptor.executable}")
        return multiprocessing.Process(
            target=AsyncBackendTVM._worker_action,
            args=(descriptor, )
        )

    def load(self, model_path: str, inputs=None, outputs=None) -> PoolBackendTVM:
        print(f"AsyncBackendTVM::load[{os.getpid()}]: ", model_path)
        for i in range(self.num_processes):
            worker_descriptor = WorkerDescriptor()
            worker_descriptor.init_executable(
                model_path, self.max_batchsize, inputs, outputs)
            self.workers.append(
                AsyncBackendTVM._create_worker(worker_descriptor))

        for worker in self.workers:
            worker.start()

        return self

    def predict(self, data: typing.Any) -> None:
        print(f"AsyncBackendTVM::predict[{os.getpid()}]: {data}")
        # Async predict, no response now
        data_id = data
        AsyncBackendTVM.concurrent_queue.put((data_id, data), block=True)

    def finish(self) -> None:
        print(f"AsyncBackendTVM::finish[{os.getpid()}]")
        for worker in self.workers:
            worker.join()

    def async_response(self) -> typing.List[typing.Any]:
        print(f"AsyncBackendTVM::async_response[{os.getpid()}]")
        result: typing.List[typing.Any] = [None] * len(self.response_map)
        for key, value in self.response_map.items():
            result[key] = value
        return result
