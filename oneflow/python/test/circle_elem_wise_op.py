import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float32)


@flow.function
def train(
    mask_incorrect=flow.input_blob_def(
        shape=(3, 3), dtype=flow.float32, is_dynamic=False
    ),
    gt_labels=flow.input_blob_def(shape=(128,), dtype=flow.float32, is_dynamic=False),
):
    mask_incorrect_num = flow.math.reduce_sum(
        flow.cast(mask_incorrect, dtype=flow.float)
    )
    # elem_cnt = flow.elem_cnt(gt_labels, dtype=flow.float) * (14.0 * 14.0)
    elem_cnt = flow.math.reduce_sum(gt_labels)
    one = flow.constant_scalar(value=1.0, dtype=flow.float)
    print(elem_cnt.has_batch_axis(), one.has_batch_axis())
    mask_accuracy = 1 - (mask_incorrect_num / flow.keras.maths.max(elem_cnt, one))


train(np.ndarray([3, 3]).astype(np.float32), np.ndarray([128]).astype(np.float32))
