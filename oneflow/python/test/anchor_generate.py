import oneflow as of
import numpy as np

image_file = "image.(2, 800, 1216, 3).npy"

anchors_files = [
    "anchors.image0.layer1.(182400, 4).npy",
    "anchors.image0.layer2.(45600, 4).npy",
    "anchors.image0.layer3.(11400, 4).npy",
    "anchors.image0.layer4.(2850, 4).npy",
    "anchors.image0.layer5.(741, 4).npy",
]

feature_map_stride = [4, 8, 16, 32, 64]
anchor_scales = [32, 64, 128, 256, 512]
aspect_ratios = [0.5, 1.0, 2.0]

OF_FUNC_TEMPLATE = """
@of.function
def anchor_generate_func_{index}(
    image_blob=of.input_blob_def({shape}, dtype=of.float)
):
    return of.detection.anchor_generate(
        image_blob,
        feature_map_stride[{index}],
        aspect_ratios,
        anchor_scales[{index}]
    )
"""

OF_CALL_OF_FUNC = """
anchor_generate_func_{index}(image).get().ndarray()
"""


def test():
    image = np.load(image_file)
    for i in range(len(anchors_files)):
        exec(OF_FUNC_TEMPLATE.format(index=i, shape=str(image.shape)))

    for i, anchors_f in enumerate(anchors_files):
        exp_anchors = np.load(anchors_f)
        anchors = eval(OF_CALL_OF_FUNC.format(index=i))
        assert np.allclose(anchors, exp_anchors)
        print("{} layer anchor_generate result: \n".format(i), anchors)


if __name__ == "__main__":
    test()
