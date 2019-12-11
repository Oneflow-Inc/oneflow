from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.record.image_pb2 as image_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.data.data_pb2 as data_util

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

    def __init__(self, auto_zero_padding=False):
        self.auto_zero_padding = auto_zero_padding

    def to_proto(self, proto=None):
        if proto is None:
            proto = op_conf_util.EncodeConf()

        proto.raw.dim1_varying_length = False
        proto.raw.auto_zero_padding = self.auto_zero_padding
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
def decode_ofrecord(
    ofrecord_dir,
    blobs,
    batch_size=1,
    data_part_num=-1,
    part_name_prefix="part-",
    part_name_suffix_length=-1,
    name=None,
):
    if name is None:
        name = id_util.UniqueStr("Decode_")

    lbis = []

    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name

    op_conf.decode_ofrecord_conf.data_dir = ofrecord_dir
    op_conf.decode_ofrecord_conf.data_part_num = data_part_num
    op_conf.decode_ofrecord_conf.batch_size = batch_size
    op_conf.decode_ofrecord_conf.part_name_prefix = part_name_prefix
    op_conf.decode_ofrecord_conf.part_name_suffix_length = (
        part_name_suffix_length
    )
    for blob_conf in blobs:
        op_conf.decode_ofrecord_conf.blob.extend([blob_conf.to_proto()])
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = name
        lbi.blob_name = blob_conf.name
        lbis.append(lbi)

    compile_context.CurJobAddOp(op_conf)
    return tuple(map(lambda x: remote_blob_util.RemoteBlob(x), lbis))


@oneflow_export("data.decode_random")
def decode_random(
    shape, dtype, batch_size=1, initializer=None, tick=None, name=None
):
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
            flow.random_uniform_initializer()
        )

    if tick:
        op_conf.decode_random_conf.tick = tick.logical_blob_name
    op_conf.decode_random_conf.out = "out"

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"

    compile_context.CurJobAddOp(op_conf)
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("data.COCODataset")
class COCODataset(object):
    def __init__(
        self,
        dataset_dir,
        annotation_file,
        image_dir,
        random_seed,
        shuffle=True,
        group_by_aspect_ratio=True,
        remove_images_without_annotations=True,
        max_segm_poly_points_per_image=65536,
        name=None,
    ):
        self.name = name or id_util.UniqueStr("COCODataset_")
        self.dataset_dir = dataset_dir
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.group_by_aspect_ratio = group_by_aspect_ratio
        self.remove_images_without_annotations = remove_images_without_annotations
        self.max_segm_poly_points_per_image = max_segm_poly_points_per_image

    def to_proto(self, proto=None):
        if proto is None:
            proto = data_util.DatasetProto()

        proto.name = self.name
        proto.dataset_dir = self.dataset_dir
        proto.shuffle = self.shuffle
        proto.random_seed = self.random_seed
        proto.coco.annotation_file = self.annotation_file
        proto.coco.image_dir = self.image_dir
        proto.coco.group_by_aspect_ratio = self.group_by_aspect_ratio
        proto.coco.remove_images_without_annotations = self.remove_images_without_annotations
        proto.coco.max_segm_poly_points = self.max_segm_poly_points_per_image
        return proto


@oneflow_export("data.TargetResizeTransform")
class TargetResizeTransform(object):
    def __init__(self, target_size, max_size):
        self.target_size = target_size
        self.max_size = max_size

    def to_proto(self, proto=None):
        if proto is None:
            proto = data_util.DataTransformProto()

        proto.target_resize.target_size = self.target_size
        proto.target_resize.max_size = self.max_size
        return proto


@oneflow_export("data.SegmentationPolygonListToMask")
class SegmentationPolygonListToMask(object):
    def to_proto(self, proto=None):
        if proto is None:
            proto = data_util.DataTransformProto()

        proto.segmentation_poly_to_mask.SetInParent()
        return proto


@oneflow_export("data.SegmentationPolygonListToAlignedMask")
class SegmentationPolygonListToAlignedMask(object):
    def to_proto(self, proto=None):
        if proto is None:
            proto = data_util.DataTransformProto()

        proto.segmentation_poly_to_aligned_mask.SetInParent()
        return proto


@oneflow_export("data.ImageNormalizeByChannel")
class ImageNormalizeByChannel(object):
    r"""note: normalize by channel, channel color space is BGR"""

    def __init__(self, mean, std=1.0):
        if isinstance(mean, (int, float)):
            mean = (float(mean), float(mean), float(mean))

        if isinstance(std, (int, float)):
            std = (float(std), float(std), float(std))

        assert isinstance(mean, (tuple, list))
        assert isinstance(std, (tuple, list))
        assert len(mean) == len(std)

        self.mean = mean
        self.std = std

    def to_proto(self, proto=None):
        if proto is None:
            proto = data_util.DataTransformProto()

        proto.image_normalize_by_channel.mean.extend(list(self.mean))
        proto.image_normalize_by_channel.std.extend(list(self.std))
        return proto


@oneflow_export("data.ImageRandomFlip")
class ImageRandomFlip(object):
    r"""Random flip image.
    params:
        @flip_code:
            0 means flipping vertically as also as flipping around the horizontal axis
            >= 1 means flipping horizontally as also as flipping around the vertical axis
            <= -1 means flipping around both axes
        @probability: probability of random flip image
    """
    def __init__(self, flip_code=1, probability=0.5):
        self.flip_code = flip_code
        self.probability = probability

    def to_proto(self, proto=None):
        if proto is None:
            proto = data_util.DataTransformProto()

        proto.image_random_flip.flip_code = self.flip_code
        proto.image_random_flip.probability = self.probability
        return proto


@oneflow_export("data.ImageAlign")
class ImageAlign(object):
    def __init__(self, alignment):
        self.alignment = alignment

    def to_proto(self, proto=None):
        if proto is None:
            proto = data_util.DataTransformProto()

        proto.image_align.alignment = self.alignment
        return proto


@oneflow_export("data.DataLoader")
class DataLoader(object):
    def __init__(self, dataset, batch_size, batch_cache_size):
        self._dataset = dataset
        self._batch_size = batch_size
        self._batch_cache_size = batch_cache_size
        self._blobs = []
        self._transforms = []

    def __call__(self, name):
        assert hasattr(
            self, "_outputs"
        ), "Call DataLoader.init first before get blob"
        return self._outputs[name]

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs):
        self._batch_size = bs

    def add_blob(
        self,
        name,
        data_source,
        shape,
        dtype,
        variable_length_axes=None,
        is_dynamic=False,
    ):
        variable_length_axes = variable_length_axes or []
        assert isinstance(variable_length_axes, (tuple, list))
        self._blobs.append(
            dict(
                name=name,
                data_source=data_source,
                shape=shape,
                dtype=dtype,
                variable_length_axes=variable_length_axes or [],
                is_dynamic=is_dynamic,
            )
        )

    def add_transform(self, transform):
        if isinstance(transform, SegmentationPolygonListToAlignedMask):
            if not any([isinstance(t, ImageAlign) for t in self._transforms]):
                raise ValueError(
                    "Need do ImageAlign before SegmentationPolygonListToAlignedMask"
                )

        self._transforms.append(transform)

    def init(self, name=None):
        if name is None:
            name = id_util.UniqueStr("DataLoad_")
        assert isinstance(name, str)

        target_resize_order = -1
        image_align_order = -1
        for i, transform in enumerate(self._transforms):
            if isinstance(transform, TargetResizeTransform):
                target_resize_order = i

            if isinstance(transform, ImageAlign):
                image_align_order = i

        if target_resize_order >= 0 and target_resize_order > image_align_order:
            raise ValueError("Need do ImageAlign after TargetResizeTransform")

        self._outputs = {}

        op_conf = op_conf_util.OperatorConf()
        op_conf.name = name
        op_conf.data_load_conf.batch_size = self._batch_size
        op_conf.data_load_conf.batch_cache_size = self._batch_cache_size
        self._dataset.to_proto(op_conf.data_load_conf.dataset)
        op_conf.data_load_conf.transforms.extend(
            [transform.to_proto() for transform in self._transforms]
        )
        for blob in self._blobs:
            blob_conf = op_conf_util.BlobConf()
            blob_conf.name = blob["name"]
            blob_conf.data_source = blob["data_source"]
            blob_conf.shape.dim.extend(blob["shape"])
            blob_conf.data_type = blob["dtype"]
            if blob_conf.data_source == data_util.kImage:
                blob_conf.encode_case.jpeg.SetInParent()
            else:
                blob_conf.encode_case.raw.SetInParent()
            blob_conf.variable_length_axes.extend(blob["variable_length_axes"])
            blob_conf.is_dynamic = blob["is_dynamic"]
            op_conf.data_load_conf.blobs.extend([blob_conf])

            lbi = logical_blob_id_util.LogicalBlobId()
            lbi.op_name = op_conf.name
            lbi.blob_name = blob_conf.name
            self._outputs[blob_conf.name] = remote_blob_util.RemoteBlob(lbi)

        compile_context.CurJobAddOp(op_conf)
