from oneflow.nn.graph import Graph
import oneflow as flow
from google.protobuf import text_format
from iree import runtime as ireert
from iree.compiler import compile_str
import numpy as np

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


    def __init__(self, graph: Graph, target=Cpu):
        self.graph = graph
        self.target = target
    
    def cpu(self):
        self.target = Iree.Cpu
        self.build()
    
    def cuda(self):
        self.target = Iree.Cuda
        self.build()

    def build(self):
        [step() for step in (self._get_job, self._convert_job_to_tosa, self._generate_context)]
    
    def _get_job(self):
        if not self.graph._is_compiled:
            print('graph is not compiled')
            exit(1)
        self.job = str(text_format.MessageToString(self.graph._forward_job_proto))
    
    def _convert_job_to_tosa(self):
        self.tosa = flow._oneflow_internal.nn.graph.ConvertJobToTosaIR(self.job)

    def _generate_context(self):
        self.ctx = ireert.SystemContext(config=self.target.config)
        flat_buffer = compile_str(self.tosa, target_backends=self.target.backend, input_type='tosa')
        vm_module = ireert.VmModule.from_flatbuffer(flat_buffer)
        self.ctx.add_vm_module(vm_module)


class Runner(object):

    def __init__(self, graph: Graph, backend=Iree):
        self.graph = graph
        self.backend = Iree(graph)

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
        if not self.graph._is_compiled:
            self.graph._compile(*args, **kwargs)

        self.backend.build()
        ctx = self.backend.ctx
        name = self.graph._name
        f = ctx.modules.module[name]
        return f
    
    def _parse_output(self, output):
        if output.is_host_accessible:
            return output
        else:
            return output.to_host()

    def __call__(self, *args, **kwargs):
        input = self._parse_input(*args, **kwargs)
        function = self._get_function(*args, **kwargs)
        output = function(*input)
        return self._parse_output(output)