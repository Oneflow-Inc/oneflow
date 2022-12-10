from PIL import Image
import numpy as np
import json
import os

from imagenet1000_clsidx_to_labels import clsidx_2_labels
from vision_resnet import resnet50
import oneflow as flow

np.random.seed(1)
model = resnet50(pretrained=False)
cross_entropy = flow.nn.CrossEntropyLoss(reduction="mean")


inp = np.random.randn(4,3,224,224)

state_dict = flow.hub.load_state_dict_from_url('https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet50.zip')
model.load_state_dict(state_dict)
model = model.to("npu")
model.eval()

# inputs = flow.tensor(inp,dtype=flow.float32,requires_grad=False)
# inputs_n = inputs.to("npu")


# optimizer = flow.optim.TORCH_SGD(model.parameters(), lr = 0.01, momentum=0.9)
# labels = flow.ones(4,dtype=flow.int32).to('npu')
# out = model(inputs_n)
# loss = cross_entropy(out, labels)
# loss.backward()
# optimizer.step()
# print("out.shape >>>>>>>>>> ", out.shape)
# flow.npu.synchronize()


def load_image(image_path="fish.jpg"):
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]
    im = Image.open(image_path)
    im = im.resize((224, 224))
    im = im.convert("RGB")
    im = np.array(im).astype("float32")
    im = (im - rgb_mean) / rgb_std
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, "float32")


# image_path = "fish.jpg"
image_path = "tiger.jpg"
image = load_image(image_path)
image = flow.tensor(image, dtype=flow.float32).to("npu")
pred = model(image).softmax().numpy()
prob = np.max(pred)
clsidx = np.argmax(pred)
image_cls = clsidx_2_labels[clsidx]
print(
        "predict image ({}) prob: {:.5f}, class name: {}".format(
            os.path.basename(image_path), prob, image_cls
        )
    )



