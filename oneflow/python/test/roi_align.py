import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)


@flow.function
def RoiAlignJob(x=flow.input_blob_def((2, 1024, 45, 45)), rois=flow.input_blob_def((2, 5), dtype=flow.float32)):
    return flow.detection.roi_align(x, rois, 4, 4)


fm = np.random.randn(2, 1024, 45, 45).astype(np.float32)
rois = np.array([[0, 1, 2, 3, 4], [1, 1, 2, 3, 4]], dtype=np.float32)
print(RoiAlignJob(fm, rois).get())
