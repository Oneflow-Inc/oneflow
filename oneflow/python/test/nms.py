import oneflow as of
import numpy as np

pt_input_files = [
    "proposals.image0.layer1.(2000, 4).npy",
    "proposals.image0.layer2.(2000, 4).npy",
    "proposals.image0.layer3.(2000, 4).npy",
    "proposals.image0.layer4.(2000, 4).npy",
    "proposals.image0.layer5.(663, 4).npy",
]

pt_output_files = [
    "proposals.image0.layer1.(1061, 4).npy",
    "proposals.image0.layer2.(1080, 4).npy",
    "proposals.image0.layer3.(902, 4).npy",
    "proposals.image0.layer4.(677, 4).npy",
    "proposals.image0.layer5.(162, 4).npy",
]

OF_FUNC_TEMPLATE = """
@of.function
def detection_nms_func_{layer}(
    input_blob=of.input_blob_def(input.shape, dtype=of.float)
):
    return of.detection.nms(input_blob)
"""

OF_CALL_OF_FUNC = """
detection_nms_func_{layer}(input).get().ndarray()
"""


def test():
    input_ndarray_list = []
    for i, input_f in enumerate(pt_input_files):
        input = np.load(input_f)
        exec(OF_FUNC_TEMPLATE.format(layer=i))
        input_ndarray_list.append(input)

    for i, (input, output_f) in enumerate(
        zip(input_ndarray_list, pt_output_files)
    ):
        keep_mask = eval(OF_CALL_OF_FUNC.format(layer=i))
        keep = np.where(keep_mask == 1)[0]
        assert np.allclose(np.load(output_f), input[keep])
        print(
            "{}: ndarray from '{}' pass in nms, out is allclose to ndarray from '{}', Done!".format(
                i, pt_input_files[i], output_f
            )
        )


if __name__ == "__main__":
    test()
