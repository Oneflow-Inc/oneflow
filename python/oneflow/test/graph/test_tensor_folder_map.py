import oneflow as flow
from flowvision.models.resnet import resnet18
import oneflow.nn as nn
import numpy as np
import copy
import os
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
os.environ["ONEFLOW_DEBUG_MODE"] = "1"
os.environ["ONEFLOW_MLIR_DUMPMLIR"] = "1"

def test_tensor_folder_map():
    data = flow.randn(1, 3, 224, 224)

    model = resnet18(pretrained=False, progress=True)
    model.eval()
    eager_res = model(data)
    copymodel = copy.deepcopy(model)
    param_table = dict(copymodel.named_parameters(prefix='model'))
    buffer_table = dict(copymodel.named_buffers(prefix='model'))
    param_table.update(buffer_table)

    class Resnet18Graph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model

        def build(self, *input):
            return self.model(*input)

    graph = Resnet18Graph()
    _ = graph(data)

    output_tensor_name_list, output_tensor_list = flow._oneflow_internal.DumpVariableTensorMgr()
    for input_tensor_names, output_tensor_name, eval_func in graph.tensor_folder_map.map:
        input_tensor_list = [param_table[name].data for name in input_tensor_names]
        manual_output_tensor = eval_func(*input_tensor_list)

        index = output_tensor_name_list.index(output_tensor_name)
        automatic_output_tensor = output_tensor_list[index]

        assert np.allclose(manual_output_tensor.numpy(), automatic_output_tensor.numpy(), rtol=1e-2, atol=1e-2)

        flow._oneflow_internal.FillVariableTensorMgr([output_tensor_name], [manual_output_tensor])
    
    lazy_res = graph(data)
    assert np.allclose(eager_res.numpy(), lazy_res.numpy(), rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    test_tensor_folder_map()