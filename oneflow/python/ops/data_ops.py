"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

from typing import Optional, Sequence, Tuple, Union, List

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export, oneflow_deprecate
import oneflow_api
import traceback


@oneflow_export("data.ImagePreprocessor")
class ImagePreprocessor(object):
    def __init__(self, preprocessor: str) -> None:
        assert isinstance(preprocessor, str)
        if preprocessor.lower() != "bgr2rgb" and preprocessor.lower() != "mirror":
            raise ValueError('preprocessor must be "bgr2rgb" or "mirror".')

        self.preprocessor = preprocessor

    def is_rgb(self) -> bool:
        return self.preprocessor.lower() == "bgr2rgb"

    def is_mirror(self) -> bool:
        return self.preprocessor.lower() == "mirror"


@oneflow_export("data.ImageResizePreprocessor")
class ImageResizePreprocessor(object):
    def __init__(self, width: int, height: int) -> None:
        assert isinstance(width, int)
        assert isinstance(height, int)
        self.width = width
        self.height = height


@oneflow_export("data.ImageCodec")
class ImageCodec(object):
    def __init__(
        self,
        image_preprocessors: Optional[
            Sequence[Union[ImagePreprocessor, ImageResizePreprocessor,]]
        ] = None,
    ) -> None:
        if isinstance(image_preprocessors, (list, tuple)):
            self.image_preprocessors = list(image_preprocessors)
        else:
            self.image_preprocessors = []

    def color_space(self) -> str:
        for img_preprocessor in self.image_preprocessors:
            if (
                isinstance(img_preprocessor, ImagePreprocessor)
                and img_preprocessor.is_rgb()
            ):
                return "RGB"
        return "BGR"

    def do_mirror(self) -> bool:
        for img_preprocessor in self.image_preprocessors:
            if (
                isinstance(img_preprocessor, ImagePreprocessor)
                and img_preprocessor.is_mirror()
            ):
                return True
        return False

    def do_resize(self):
        for img_preprocessor in self.image_preprocessors:
            if isinstance(img_preprocessor, ImageResizePreprocessor):
                return (True, img_preprocessor.width, img_preprocessor.height)
        return (False, -1, -1)


@oneflow_export("data.RawCodec")
class RawCodec(object):
    def __init__(self, auto_zero_padding: bool = False) -> None:
        self.auto_zero_padding = auto_zero_padding


@oneflow_export("data.NormByChannelPreprocessor")
class NormByChannelPreprocessor(object):
    def __init__(
        self,
        mean_values: Union[List[float], Tuple[float]],
        std_values: Union[List[float], Tuple[float]] = (1.0, 1.0, 1.0),
        data_format: str = "channels_last",
    ) -> None:
        assert isinstance(mean_values, (list, tuple))
        assert isinstance(std_values, (list, tuple))
        assert isinstance(data_format, str)
        self.mean_values = mean_values
        self.std_values = std_values
        self.data_format = data_format

    def output_layout(self) -> str:
        if self.data_format == "channels_last":
            return "NHWC"
        else:
            return "NCHW"


@oneflow_export("data.BlobConf")
class BlobConf(object):
    def __init__(
        self,
        name: str,
        shape: Sequence[int],
        dtype: flow.dtype,
        codec: Union[ImageCodec, RawCodec],
        preprocessors: Optional[Sequence[Union[NormByChannelPreprocessor,]]] = None,
    ) -> None:
        assert isinstance(name, str)
        assert isinstance(shape, (list, tuple))

        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.codec = codec

        if isinstance(preprocessors, (list, tuple)):
            self.preprocessors = list(preprocessors)
        else:
            self.preprocessors = []

    def decode_blob(
        self, input_blob: oneflow_api.BlobDesc, batch_size: int
    ) -> oneflow_api.BlobDesc:
        if isinstance(self.codec, ImageCodec):
            color_space = self.codec.color_space()
            image = flow.data.ofrecord_image_decoder(
                input_blob=input_blob, blob_name=self.name, color_space=color_space
            )
            coin_flip = None
            if self.codec.do_mirror():
                coin_flip = flow.random.coin_flip(batch_size)

            do_resize, width, height = self.codec.do_resize()
            if do_resize:
                assert width > 0 and height > 0
                image, _, _ = flow.image.resize(
                    image=image, target_size=(width, height)
                )
            else:
                assert len(self.shape) >= 2
                image, _, _ = flow.image.resize(
                    image=image, target_size=(self.shape[0], self.shape[1])
                )
            for preprocess in self.preprocessors:
                image = flow.image.crop_mirror_normalize(
                    input_blob=image,
                    mirror_blob=coin_flip,
                    color_space=color_space,
                    output_layout=preprocess.output_layout(),
                    mean=preprocess.mean_values,
                    std=preprocess.std_values,
                    output_dtype=self.dtype,
                )
            return image
        elif isinstance(self.codec, RawCodec):
            raw = flow.data.ofrecord_raw_decoder(
                input_blob=input_blob,
                blob_name=self.name,
                shape=self.shape,
                dtype=self.dtype,
                auto_zero_padding=self.codec.auto_zero_padding,
            )
            return raw
        else:
            raise NotImplementedError


@oneflow_export("data.decode_ofrecord")
@oneflow_deprecate()
def decode_ofrecord(
    ofrecord_dir: str,
    blobs: Sequence[BlobConf],
    batch_size: int = 1,
    data_part_num: int = 1,
    part_name_prefix: str = "part-",
    part_name_suffix_length: int = -1,
    shuffle: bool = False,
    buffer_size: int = 1024,
    name: str = None,
) -> Tuple[oneflow_api.BlobDesc]:
    print(
        "WARNING:",
        "oneflow.data.decode_ofrecord is deprecated, and NOT work in eager mode, please use: \n",
        "    1)   ofrecord = oneflow.data.ofrecord_reader(...) to read ofrecord; \n",
        "    2)   image = oneflow.data.ofrecord_image_decoder(...) to decode image; \n",
        "    3)   raw = oneflow.data.ofrecord_raw_decoder(...) to decode raw data like label; \n",
        traceback.format_stack()[-2],
    )
    assert not flow.eager_execution_enabled()

    ofrecord = flow.data.ofrecord_reader(
        ofrecord_dir=ofrecord_dir,
        batch_size=batch_size,
        data_part_num=data_part_num,
        part_name_prefix=part_name_prefix,
        part_name_suffix_length=part_name_suffix_length,
        random_shuffle=shuffle,
        shuffle_buffer_size=buffer_size,
        name=name,
    )

    result_blob_list = []
    for blob_conf in blobs:
        result_blob_list.append(
            blob_conf.decode_blob(input_blob=ofrecord, batch_size=batch_size)
        )

    return tuple(result_blob_list)


@oneflow_export("data.ofrecord_loader")
def ofrecord_loader(
    ofrecord_dir: str,
    batch_size: int = 1,
    data_part_num: int = 1,
    part_name_prefix: str = "part-",
    part_name_suffix_length: int = -1,
    shuffle: bool = False,
    shuffle_buffer_size: int = 1024,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    print(
        "WARNING:",
        "oneflow.data.ofrecord_loader is deprecated, and NOT work in eager mode, please use: \n",
        "    ofrecord = oneflow.data.ofrecord_reader(...) to read ofrecord; \n",
        traceback.format_stack()[-2],
    )

    return flow.data.ofrecord_reader(
        ofrecord_dir=ofrecord_dir,
        batch_size=batch_size,
        data_part_num=data_part_num,
        part_name_prefix=part_name_prefix,
        part_name_suffix_length=part_name_suffix_length,
        random_shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        name=name,
    )


@oneflow_export("data.ofrecord_reader")
def ofrecord_reader(
    ofrecord_dir: str,
    batch_size: int = 1,
    data_part_num: int = 1,
    part_name_prefix: str = "part-",
    part_name_suffix_length: int = -1,
    random_shuffle: bool = False,
    shuffle_buffer_size: int = 1024,
    shuffle_after_epoch: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""Get ofrecord object from ofrecord dataset.

    Args:
        ofrecord_dir (str): Path to ofrecord dataset.
        batch_size (int, optional): Batch size. Defaults to 1.
        data_part_num (int, optional): Number of dataset's partitions. Defaults to 1.
        part_name_prefix (str, optional): Prefix of dataset's parition file. Defaults to "part-".
        part_name_suffix_length (int, optional): Total length of padded suffix number , -1 means no padding. eg: 3 for `part-001`. Defaults to -1.
        random_shuffle (bool, optional): Determines records shuffled or not. Defaults to False.
        shuffle_buffer_size (int, optional): Shuffle buffer size. Defaults to 1024.
        shuffle_after_epoch (bool, optional): Shuffled or not after each epoch. Defaults to False.
        name (Optional[str], optional): Optional name. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example:

    .. code-block:: python

        import oneflow as flow
        import oneflow.typing as tp
        from typing import Tuple


        @flow.global_function(type="predict")
        def ofrecord_reader_job() -> Tuple[tp.Numpy, tp.Numpy]:
            batch_size = 16
            with flow.scope.placement("cpu", "0:0"):
                # our ofrecord file path is "./dataset/part-0"
                ofrecord = flow.data.ofrecord_reader(
                    "./dataset/",
                    batch_size=batch_size,
                    data_part_num=1,
                    part_name_suffix_length=-1,
                    part_name_prefix='part-',
                    random_shuffle=True,
                    shuffle_after_epoch=True,
                )
                # image shape is (28*28, )
                image = flow.data.OFRecordRawDecoder(
                    ofrecord, "images", shape=(784, ), dtype=flow.int32
                )
                # label shape is (1, )
                label = flow.data.OFRecordRawDecoder(
                    ofrecord, "labels", shape=(1, ), dtype=flow.int32
                )

                return image, label

        if __name__ == "__main__":
            images, labels = ofrecord_reader_job()
            print("In per batch, images shape is", images.shape)
            print("In per batch, labels shape is", labels.shape)

            # In per batch, images shape is (16, 784)
            # In per batch, labels shape is (16, 1)

    """
    if name is None:
        name = id_util.UniqueStr("OFRecord_Reader_")

    return (
        flow.user_op_builder(name)
        .Op("OFRecordReader")
        .Output("out")
        .Attr("data_dir", ofrecord_dir)
        .Attr("data_part_num", data_part_num)
        .Attr("batch_size", batch_size)
        .Attr("part_name_prefix", part_name_prefix)
        .Attr("random_shuffle", random_shuffle)
        .Attr("shuffle_buffer_size", shuffle_buffer_size)
        .Attr("shuffle_after_epoch", shuffle_after_epoch)
        .Attr("part_name_suffix_length", part_name_suffix_length)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("data.decode_random")
def decode_random(
    shape: Sequence[int],
    dtype: flow.dtype,
    batch_size: int = 1,
    initializer: Optional[initializer_conf_util.InitializerConf] = None,
    tick: Optional[oneflow_api.BlobDesc] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    op_conf = op_conf_util.OperatorConf()

    if name is None:
        name = id_util.UniqueStr("DecodeRandom_")
    assert isinstance(name, str)
    op_conf.name = name

    assert isinstance(shape, (list, tuple))
    op_conf.decode_random_conf.shape.dim.extend(shape)

    assert dtype is not None
    setattr(
        op_conf.decode_random_conf,
        "data_type",
        oneflow_api.deprecated.GetProtoDtype4OfDtype(dtype),
    )

    op_conf.decode_random_conf.batch_size = batch_size

    if initializer is not None:
        op_conf.decode_random_conf.data_initializer.CopyFrom(initializer)
    else:
        op_conf.decode_random_conf.data_initializer.CopyFrom(
            flow.random_uniform_initializer()
        )

    if tick:
        op_conf.decode_random_conf.tick = tick.unique_name
    op_conf.decode_random_conf.out = "out"

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"

    interpret_util.ConsistentForward(op_conf)
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export(
    "data.image_decoder_random_crop_resize", "data.ImageDecoderRandomCropResize"
)
def image_decoder_random_crop_resize(
    input_blob: oneflow_api.BlobDesc,
    target_width: int,
    target_height: int,
    num_attempts: Optional[int] = None,
    seed: Optional[int] = None,
    random_area: Optional[Sequence[float]] = None,
    random_aspect_ratio: Optional[Sequence[float]] = None,
    num_workers: Optional[int] = None,
    warmup_size: Optional[int] = None,
    max_num_pixels: Optional[int] = None,
    name: Optional[str] = None,
) -> Tuple[oneflow_api.BlobDesc]:
    if name is None:
        name = id_util.UniqueStr("ImageDecoderRandomCropResize_")

    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.image_decoder_random_crop_resize_conf, "in", input_blob.unique_name)
    op_conf.image_decoder_random_crop_resize_conf.out = "out"
    op_conf.image_decoder_random_crop_resize_conf.target_width = target_width
    op_conf.image_decoder_random_crop_resize_conf.target_height = target_height
    if num_attempts is not None:
        op_conf.image_decoder_random_crop_resize_conf.num_attempts = num_attempts
    if seed is not None:
        op_conf.image_decoder_random_crop_resize_conf.seed = seed
    if random_area is not None:
        assert len(random_area) == 2
        op_conf.image_decoder_random_crop_resize_conf.random_area_min = random_area[0]
        op_conf.image_decoder_random_crop_resize_conf.random_area_max = random_area[1]
    if random_aspect_ratio is not None:
        assert len(random_aspect_ratio) == 2
        op_conf.image_decoder_random_crop_resize_conf.random_aspect_ratio_min = random_aspect_ratio[
            0
        ]
        op_conf.image_decoder_random_crop_resize_conf.random_aspect_ratio_max = random_aspect_ratio[
            1
        ]
    if num_workers is not None:
        op_conf.image_decoder_random_crop_resize_conf.num_workers = num_workers
    if warmup_size is not None:
        op_conf.image_decoder_random_crop_resize_conf.warmup_size = warmup_size
    if max_num_pixels is not None:
        op_conf.image_decoder_random_crop_resize_conf.max_num_pixels = max_num_pixels
    interpret_util.Forward(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("data.onerec_reader")
def onerec_reader(
    files,
    batch_size=1,
    random_shuffle=False,
    shuffle_mode="instance",
    shuffle_buffer_size=1024,
    shuffle_after_epoch=False,
    verify_example=True,
    name=None,
):
    assert isinstance(files, (list, tuple))

    if name is None:
        name = id_util.UniqueStr("OneRecReader_")

    return (
        flow.user_op_builder(name)
        .Op("OneRecReader")
        .Output("out")
        .Attr("files", files)
        .Attr("batch_size", batch_size)
        .Attr("random_shuffle", random_shuffle)
        .Attr("shuffle_mode", shuffle_mode)
        .Attr("shuffle_buffer_size", shuffle_buffer_size)
        .Attr("shuffle_after_epoch", shuffle_after_epoch)
        .Attr("verify_example", verify_example)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
