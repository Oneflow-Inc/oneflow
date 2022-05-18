from oneflow.nn.graph import Graph
import oneflow as flow
from google.protobuf import text_format
from iree import runtime as ireert
from iree.compiler import compile_str

class BackBend():
    IREE = 'iree'

class VmFunction(object):
    def __init__(self, graph: Graph, backend: BackBend=BackBend.IREE):
        self.graph = graph
        self.backend = backend


    def __call__(self, *args, **kwargs):
        if not self.graph._is_compiled:
            self.graph._compile(*args, **kwargs)
        
        serialized_job = str(text_format.MessageToString(self.graph._forward_job_proto))
        self.tosa = flow._oneflow_internal.nn.graph.ConvertJobToTosaIR(serialized_job)
        return self._run_tosa_on_iree(*args, **kwargs)
    
    def _run_tosa_on_iree(self,  *args, **kwargs):
        compiled_flatbuffer = compile_str(self.tosa, target_backends=['dylib-llvm-aot'], input_type='tosa')
        vm_module = ireert.VmModule.from_flatbuffer(compiled_flatbuffer)
        config = ireert.Config('dylib')
        ctx = ireert.SystemContext(config=config)
        ctx.add_vm_module(vm_module)
        name = self.graph._name
        f = ctx.modules.module[name]
        input = args[0]
        res =  f(input.cpu().detach().numpy())
        return res.to_host()