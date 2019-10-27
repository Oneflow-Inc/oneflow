from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.record.image_pb2 as image_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("data.ImagePreprocessor")
class ImagePreprocessor(object):
    def __init__(self, preprocessor):
        assert isinstance(preprocessor, str)
        if (
            preprocessor.lower() != "bgr2rgb"
            and preprocessor.lower() != "mirror"
        ):
            raise ValueError('preprocessor must be "bgr2rgb" or "mirror".')

        self.preprocessor = preprocessor

    def to_proto(self, proto=None):
        if proto is None:
            proto = image_util.ImagePreprocess()

        if self.preprocessor == "bgr2rgb":
            proto.bgr2rgb.SetInParent()
        elif self.preprocessor == "mirror":
            proto.mirror.SetInParent()
        else:
            raise NotImplementedError

        return proto


@oneflow_export("data.ImageResizePreprocessor")
class ImageResizePreprocessor(object):
    def __init__(self, width, height):
        assert isinstance(width, int)
        assert isinstance(height, int)

        self.width = width
        self.height = height

    def to_proto(self, proto=None):
        proto = proto or image_util.ImagePreprocess()
        setattr(proto.resize, "width", self.width)
        setattr(proto.resize, "height", self.height)
        return proto


@oneflow_export("data.ImageCropPreprocessor")
class ImageCropPreprocessor(object):
    def __init__(self, width, height, random_xy=True, x=None, y=None, crop_w=-1, crop_h=-1):
        assert isinstance(width, int)
        assert isinstance(height, int)
        self.random_xy = random_xy
        self.x = None
        self.y = None
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        self.width = width
        self.height = height
        self.crop_h = crop_h
        self.crop_w = crop_w

    def to_proto(self, proto=None):
        proto = proto or image_util.ImagePreprocess()
        proto.crop.random_xy = self.random_xy
        if self.x is not None:
            proto.crop.x = self.x
        if self.y is not None:
            proto.crop.y = self.y
        proto.crop.width = self.width
        proto.crop.height = self.height
        proto.crop.range.width = self.crop_w
        proto.crop.range.height = self.crop_h
        return proto


@oneflow_export("data.ImageCutoutPreprocessor")
class ImageCutoutPreprocessor(object):
    def __init__(self, cutout_ratio = 0.0, cutout_size = 4, cutout_mode = "normal", cutout_filler = 0):
        self.cutout_ratio = cutout_ratio
        self.cutout_size = cutout_size
        self.cutout_mode = cutout_mode
        self.cutout_filler = cutout_filler

    def to_proto(self, proto=None):
        proto = proto or image_util.ImagePreprocess()
        proto.cutout.cutout_ratio = self.cutout_ratio
        proto.cutout.cutout_size = self.cutout_size
        if self.cutout_mode == "normal":
            proto.cutout.cutout_mode = image_util.ImageCutout.CutoutMode.Normal
        else:
            proto.cutout.cutout_mode = image_util.ImageCutout.CutoutMode.Uniform
        proto.cutout.cutout_filler = self.cutout_filler
        return proto


@oneflow_export("data.ImageCodec")
class ImageCodec(object):
    def __init__(self, image_preprocessors=None):
        if isinstance(image_preprocessors, (list, tuple)):
            self.image_preprocessors = list(image_preprocessors)
        else:
            self.image_preprocessors = []

    def to_proto(self, proto=None):
        if proto is None:
            proto = op_conf_util.EncodeConf()

        proto.jpeg.preprocess.extend(
            [p.to_proto() for p in self.image_preprocessors]
        )
        return proto


@oneflow_export("data.RawCodec")
class RawCodec(object):
    def to_proto(self, proto=None):
        if proto is None:
            proto = op_conf_util.EncodeConf()

        proto.raw.dim1_varying_length = False
        return proto


@oneflow_export("data.NormByChannelPreprocessor")
class NormByChannelPreprocessor(object):
    def __init__(
        self,
        mean_values,
        std_values=(1.0, 1.0, 1.0),
        data_format="channels_last",
    ):
        assert isinstance(mean_values, (list, tuple))
        assert isinstance(std_values, (list, tuple))
        assert isinstance(data_format, str)
        self.mean_values = mean_values
        self.std_values = std_values
        self.data_format = data_format

    def to_proto(self, proto=None):
        if proto is None:
            proto = op_conf_util.PreprocessConf()

        proto.norm_by_channel_conf.mean_value.extend(self.mean_values)
        proto.norm_by_channel_conf.std_value.extend(self.std_values)
        proto.norm_by_channel_conf.data_format = self.data_format

        return proto


@oneflow_export("data.BlobConf")
class BlobConf(object):
    def __init__(self, name, shape, dtype, codec, preprocessors=None):
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

    def to_proto(self):
        blob_conf = op_conf_util.BlobConf()
        blob_conf.name = self.name
        blob_conf.shape.dim.extend(self.shape)
        blob_conf.data_type = self.dtype
        self.codec.to_proto(blob_conf.encode_case)
        blob_conf.preprocess.extend([p.to_proto() for p in self.preprocessors])
        return blob_conf


@oneflow_export("data.decode_ofrecord")
def decode_ofrecord(ofrecord_dir, blobs,
                    batch_size=1,
                    data_part_num=-1,
                    part_name_prefix="part-",
                    part_name_suffix_length=-1,
                    shuffle=False,
                    buffer_size=1024,
                    name=None):
    if name is None:
        name = id_util.UniqueStr("Decode_")

    lbis = []

    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name

    op_conf.decode_ofrecord_conf.data_dir = ofrecord_dir
    op_conf.decode_ofrecord_conf.data_part_num = data_part_num
    op_conf.decode_ofrecord_conf.batch_size = batch_size
    op_conf.decode_ofrecord_conf.part_name_prefix = part_name_prefix
    op_conf.decode_ofrecord_conf.part_name_suffix_length = part_name_suffix_length
    if shuffle == True:
        op_conf.decode_ofrecord_conf.random_shuffle_conf.buffer_size = buffer_size
    for blob_conf in blobs:
        op_conf.decode_ofrecord_conf.blob.extend([blob_conf.to_proto()])
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = name
        lbi.blob_name = blob_conf.name
        lbis.append(lbi)

    compile_context.CurJobAddOp(op_conf)
    return tuple(map(lambda x: remote_blob_util.RemoteBlob(x), lbis))

@oneflow_export("data.decode_random")
def decode_random(shape, dtype, batch_size=1, initializer=None, tick=None, name=None):
    op_conf = op_conf_util.OperatorConf()

    if name is None:
        name = id_util.UniqueStr("DecodeRandom_")
    assert isinstance(name, str)
    op_conf.name = name

    assert isinstance(shape, (list, tuple))
    op_conf.decode_random_conf.shape.dim.extend(shape)

    assert dtype is not None
    setattr(op_conf.decode_random_conf, "data_type", dtype)

    op_conf.decode_random_conf.batch_size = batch_size

    if initializer is not None:
        op_conf.decode_random_conf.data_initializer.CopyFrom(initializer)
    else:
        op_conf.decode_random_conf.data_initializer.CopyFrom(
                flow.random_uniform_initializer())

    if tick:
        op_conf.decode_random_conf.tick = tick.logical_blob_name
    op_conf.decode_random_conf.out = "out"

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"

    compile_context.CurJobAddOp(op_conf)
    return remote_blob_util.RemoteBlob(lbi)
