import oneflow as flow
from oneflow import nn
from flowvision.models import resnet50
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

# 模型参数存储目录
MODEL_PARAMS = 'checkpoints/resnet50'

# # 下载预训练模型并保存
# model = resnet50(pretrained=True)
# flow.save(model.state_dict(), MODEL_PARAMS)

class ResNet50Graph(nn.Graph):
    def __init__(self, eager_model):
        super().__init__()
        self.model = eager_model

    def build(self, x):
        return self.model(x)


params = flow.load(MODEL_PARAMS)
model = resnet50()
model.load_state_dict(params)

# 将模型设置为 eval 模式
model.eval()

resnet50_graph = ResNet50Graph(model)
# 构建出静态图模型
resnet50_graph._compile(flow.randn(1, 3, 224, 224))

# 导出为 ONNX 模型并进行检查
convert_to_onnx_and_check(resnet50_graph, 
                          flow_weight_dir=MODEL_PARAMS, 
                          opset=11,
                          onnx_model_path="./", 
                          print_outlier=True,
                          dynamic_batch_size=True)