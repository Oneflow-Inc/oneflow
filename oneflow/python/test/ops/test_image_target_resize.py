import cv2
import numpy as np
import oneflow as flow
from PIL import Image


def _of_image_target_resize(images, image_static_shape, target_size, max_size):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.global_function(func_config)
    def image_target_resize_job(
        images_def=flow.MirroredTensorListDef(
            shape=image_static_shape, dtype=flow.float
        )
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
        resized_images_buffer, size, scale = flow.image_target_resize(
            images_buffer, target_size, max_size
        )
        resized_images = flow.tensor_buffer_to_tensor_list(
            resized_images_buffer,
            shape=(target_size, max_size, image_static_shape[-1]),
            dtype=flow.float,
        )
        return resized_images, size, scale

    resized_images, size, scale = image_target_resize_job([images]).get()
    resized_images = resized_images.ndarray_lists()[0]
    size = size.ndarray_list()[0]
    scale = scale.ndarray_list()[0]
    return resized_images, size, scale


def _read_images_by_pil(image_files):
    images = [Image.open(image_file) for image_file in image_files]
    # convert image to BGR
    converted_images = [
        np.array(image).astype(np.single)[:, :, ::-1] for image in images
    ]
    return [np.expand_dims(image, axis=0) for image in converted_images]


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


def _target_resize_by_cv(images, target_size, max_size):
    resized_images = []
    for image in images:
        squeeze_image = image.squeeze()
        w, h = _get_target_resize_size(
            squeeze_image.shape[1], squeeze_image.shape[0], target_size, max_size
        )
        resized_images.append(cv2.resize(squeeze_image, (w, h)))

    return resized_images


def _get_target_resize_size(w, h, target_size, max_size):
    min_original_size = float(min((w, h)))
    max_original_size = float(max((w, h)))

    min_resized_size = target_size
    max_resized_size = int(
        round(max_original_size / min_original_size * min_resized_size)
    )
    if max_resized_size > max_size:
        max_resized_size = max_size
        min_resized_size = int(
            round(max_resized_size * min_original_size / max_original_size)
        )

    return (
        (min_resized_size, max_resized_size)
        if w < h
        else (max_resized_size, min_resized_size)
    )


def _compare_image_target_resize_with_cv(
    test_case, image_files, target_size, max_size, print_debug_info=False
):
    images = _read_images_by_cv(image_files)
    image_static_shape = _get_images_static_shape(images)

    resized_images, size, scale = _of_image_target_resize(
        images, tuple(image_static_shape), target_size, max_size
    )

    cv_resized_images = _target_resize_by_cv(images, target_size, max_size)

    for resized_image, cv_resized_image, image_size, image_scale in zip(
        resized_images, cv_resized_images, size, scale
    ):
        if print_debug_info:
            print("resized_image shape:", resized_image.shape)
            print("cv_resized_image shape:", cv_resized_image.shape)
            print("resized h & w:", image_size)
            print("resize h_scale & w_scale:", image_scale)

        test_case.assertTrue(np.allclose(resized_image, cv_resized_image))


def test_image_target_resize(test_case):
    _compare_image_target_resize_with_cv(
        test_case,
        [
            "/dataset/mscoco_2017/val2017/000000000139.jpg",
            "/dataset/mscoco_2017/val2017/000000000632.jpg",
        ],
        800,
        1333,
        # True,
    )
