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
        config = 'dylib'

    class Cuda(Target):
        backend = ['cuda']
        config = 'cuda'

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
    
    def cuda(self):
        self.target = Iree.Cuda

    def generate_tosa(self, graph: Graph):
        self.graph = graph
        [step() for step in (self._get_job, self._convert_job_to_tosa)]
    
    def _get_job(self):
        self.job = str(text_format.MessageToString(self.graph._forward_job_proto))
    
    def _convert_job_to_tosa(self):
        self.tosa = flow._oneflow_internal.nn.graph.ConvertJobToTosaIR(self.job)

    def generate_context(self):
        config = ireert.Config(self.target.config)
        self.ctx = ireert.SystemContext(config=config)
        flat_buffer = compile_str(self.tosa, target_backends=self.target.backend, input_type='tosa')
        vm_module = ireert.VmModule.from_flatbuffer(flat_buffer)
        self.ctx.add_vm_module(vm_module)
        return self.ctx



class Runner(object):

    _tosa_cache = {}

    def __init__(self, raw_graph, backend=Iree):
        self.raw_graph = raw_graph
        self.backend = Iree()
    
    def cuda(self):
        self.backend.cuda()
        return self

    def cpu(self):
        self.backend.cpu()
        return self

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
        full_name = self._full_name()
        if not full_name in Runner._tosa_cache:
            graph = self.raw_graph()
            # graph.build_graph(*args, **kwargs)
            graph._compile(*args, **kwargs)
            self.backend.generate_tosa(graph)
            Runner._tosa_cache[full_name] = {"name":graph._name, "data":self.backend.tosa}

        config = Runner._tosa_cache[full_name]
        self.backend.tosa = config["data"]

        ctx = self.backend.generate_context()
        f = ctx.modules.module[config["name"]]
        return f
    
    def _parse_output(self, output):
        if output.is_host_accessible:
            return output
        else:
            return output.to_host()

    def _full_name(self):
        full_name = self.raw_graph.__name__
        for elem in self.input:
            full_name += str(elem.shape) + str(elem.dtype)
        return full_name


    def __call__(self, *args, **kwargs):
        self.input = self._parse_input(*args, **kwargs)
        function = self._get_function(*args, **kwargs)
        output = function(*self.input)
        return self._parse_output(output)