import cv2
import numpy as np
import oneflow as flow


def _of_image_flip(images, image_shape, flip_code):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.global_function(func_config)
    def image_flip_job(
        images_def=flow.MirroredTensorListDef(shape=image_shape, dtype=flow.float)
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
        flip_images = flow.image_flip(images_buffer, flip_code)
        return flow.tensor_buffer_to_tensor_list(
            flip_images, shape=image_shape[1:], dtype=flow.float
        )

    image_tensor = image_flip_job([images]).get()
    return image_tensor.ndarray_lists()[0]


def _read_images_by_cv(image_files):
    images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
    return [np.expand_dims(image, axis=0) for image in images]


def _get_images_static_shape(images):
    image_shapes = [image.shape for image in images]
    image_static_shape = np.amax(image_shapes, axis=0)
    assert isinstance(
        image_static_shape, np.ndarray
    ), "image_shapes: {}, image_static_shape: {}".format(
        str(image_shapes), str(image_static_shape)
    )
    image_static_shape = image_static_shape.tolist()
    assert image_static_shape[0] == 1, str(image_static_shape)
    image_static_shape[0] = len(image_shapes)
    return image_static_shape


def _compare_image_flip_with_cv(test_case, image_files):
    images = _read_images_by_cv(image_files)
    assert all([len(image.shape) == 4 for image in images])
    image_shape = _get_images_static_shape(images)

    flip_images = _of_image_flip(images, tuple(image_shape), 1)
    for image, flip_image in zip(images, flip_images):
        exp_flip_image = cv2.flip(image.squeeze(), 1)
        test_case.assertTrue(np.allclose(exp_flip_image, flip_image))


def test_image_flip(test_case):
    _compare_image_flip_with_cv(
        test_case,
        [
            "/dataset/mscoco_2017/val2017/000000000139.jpg",
            "/dataset/mscoco_2017/val2017/000000000632.jpg",
        ],
    )
