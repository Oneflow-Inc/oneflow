import oneflow as flow
import numpy as np

from termcolor import colored

test_data_files_1 = {
    "masks": ["segmentation_mask_0.image0.(873, 800).npy"],
    "rois": ["segmentation_proposal_0.image0.(4,).npy"],
    "output": ["segmentation_resized_mask_0.image0.(28, 28).npy"],
}


@flow.function
def test_func_1(
    masks_blob=flow.input_blob_def(shape=(1, 1, 873, 800), dtype=flow.int8),
    rois_blob=flow.input_blob_def(shape=(1, 4), dtype=flow.float32),
):
    return flow.detection.masks_crop_and_resize(masks_blob, rois_blob, 28, 28)


def test_case_1():
    masks_list = []
    for mask_f in test_data_files_1["masks"]:
        masks_list.append(
            np.expand_dims(np.load(mask_f).astype(np.int8), axis=0)
        )

    masks = np.stack(masks_list, axis=0)

    rois_list = []
    for roi_f in test_data_files_1["rois"]:
        rois_list.append(np.load(roi_f))

    rois = np.stack(rois_list, axis=0)

    outputs_list = []
    for output_f in test_data_files_1["output"]:
        outputs_list.append(np.load(output_f))

    exp_output = np.stack(outputs_list, axis=0)

    output = test_func_1(masks, rois).get().ndarray()
    output = output.squeeze(1)

    print(colored("test case 1 Running", "yellow"))
    if np.allclose(output, exp_output):
        print(colored("output allclose", "green"))
    else:
        print(colored("output not close", "red"))
        print("output shape: ", output.shape)
        print("expected output shape: ", exp_output.shape)
        print("max diff: ", np.max(output - exp_output))
        np.save("output", output)
        np.save("exp_output", exp_output)

    print(colored("test case 1 Done!", "yellow"))


if __name__ == "__main__":
    test_case_1()
