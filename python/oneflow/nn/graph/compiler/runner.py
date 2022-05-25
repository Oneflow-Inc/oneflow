from oneflow.nn.graph import Graph
import oneflow as flow
from google.protobuf import text_format
from iree import runtime as ireert
from iree.compiler import compile_str
import numpy as np
import copy

class Iree:

    class Target:
        pass
    class Cpu(Target):
        backend = ['dylib-llvm-aot']
        config = ireert.Config('dylib')

    class Cuda(Target):
        backend = ['cuda']
        config = ireert.Config('dylib')

    # members
    graph: Graph
    target: Target
    job: str
    tosa: str
    ctx: ireert.SystemContext


    def __init__(self, target=Cpu):
        self.target = target
    
    def cpu(self):
        self.target = Iree.Cpu
        self.build()
    
    def cuda(self):
        self.target = Iree.Cuda
        self.build()

    def build(self, graph: Graph):
        self.graph = graph
        [step() for step in (self._get_job, self._convert_job_to_tosa, self._generate_context)]
    
    def _get_job(self):
        self.job = str(text_format.MessageToString(self.graph._forward_job_proto))
    
    def _convert_job_to_tosa(self):
        self.tosa = flow._oneflow_internal.nn.graph.ConvertJobToTosaIR(self.job)

    def _generate_context(self):
        self.ctx = ireert.SystemContext(config=self.target.config)
        flat_buffer = compile_str(self.tosa, target_backends=self.target.backend, input_type='tosa')
        vm_module = ireert.VmModule.from_flatbuffer(flat_buffer)
        self.ctx.add_vm_module(vm_module)


class Runner(object):

    _cache = {}

    def __init__(self, raw_graph, backend=Iree):
        self.raw_graph = raw_graph
        self.backend = Iree()

    def _parse_input(self, *args, **kwargs):
        res = []
        for arg in args:
            if isinstance(arg, flow.Tensor):
                res.append(arg.cpu().detach().numpy())
            elif isinstance(arg, np.ndarray):
                res.append(arg)
            else:
                print('not support class')
                exit(1)
        return res

    def _get_function(self, *args, **kwargs):
        full_name = self.full_name()
        if full_name in Runner._cache:
            return Runner._cache[full_name]
        else:
            graph = self.raw_graph()
            # graph.build_graph(*args, **kwargs)
            graph._compile(*args, **kwargs)
            self.backend.build(graph)
            ctx = self.backend.ctx
            name = graph._name
            f = ctx.modules.module[name]
            Runner._cache[full_name] = f
            return f
    
    def _parse_output(self, output):
        if output.is_host_accessible:
            return output
        else:
            return output.to_host()

    def full_name(self):
        full_name = self.raw_graph.__name__
        for elem in self.input:
            full_name += str(elem.shape) + str(elem.dtype)
        return full_name


    def __call__(self, *args, **kwargs):
        self.input = self._parse_input(*args, **kwargs)
        function = self._get_function(*args, **kwargs)
        output = function(*self.input)
        return self._parse_output(output)