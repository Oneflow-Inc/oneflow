from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.record.image_pb2 as image_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.remote_blob import BlobDef


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


@oneflow_export("data.BytesListCodec")
class BytesListCodec(object):

    def __init__(self):
        pass

    def to_proto(self, proto=None):
        if proto is None:
            proto = op_conf_util.EncodeConf()

        proto.bytes_list.SetInParent()
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
    data_part_num=1,
    part_name_prefix="part-",
    part_name_suffix_length=-1,
    shuffle=False,
    buffer_size=1024,
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
    if shuffle == True:
        op_conf.decode_ofrecord_conf.random_shuffle_conf.buffer_size = buffer_size
    for blob_conf in blobs:
        op_conf.decode_ofrecord_conf.blob.extend([blob_conf.to_proto()])
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = name
        lbi.blob_name = blob_conf.name
        lbis.append(lbi)

    compile_context.CurJobAddConsistentOp(op_conf)
    return tuple(map(lambda x: remote_blob_util.RemoteBlob(x), lbis))

@oneflow_export("data.ofrecord_loader")
def ofrecord_loader(
    ofrecord_dir,
    batch_size=1,
    data_part_num=1,
    part_name_prefix="part-",
    part_name_suffix_length=-1,
    shuffle=False,
    shuffle_buffer_size=1024,
    name=None,
):
    if name is None:
        name = id_util.UniqueStr("OFRecord_Loader_")

    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name

    op_conf.record_load_conf.out = "out"
    op_conf.record_load_conf.data_dir = ofrecord_dir
    op_conf.record_load_conf.data_part_num = data_part_num
    op_conf.record_load_conf.batch_size = batch_size
    op_conf.record_load_conf.part_name_prefix = part_name_prefix
    if part_name_suffix_length is not -1:
        op_conf.record_load_conf.part_name_suffix_length = (
            part_name_suffix_length
        )
    if shuffle:
        op_conf.record_load_conf.random_shuffle_conf.buffer_size = shuffle_buffer_size
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = name
    lbi.blob_name = "out"

    compile_context.CurJobAddConsistentOp(op_conf)
    return remote_blob_util.RemoteBlob(lbi)


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

    compile_context.CurJobAddConsistentOp(op_conf)
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
        max_segm_poly_points=1024,
        name=None,
    ):
        name = name or id_util.UniqueStr("COCODataset_")
        self.dataset_dir = dataset_dir
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.group_by_aspect_ratio = group_by_aspect_ratio
        self.max_segm_poly_points = max_segm_poly_points
        self.name = name

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
        proto.coco.max_segm_poly_points = self.max_segm_poly_points
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

    Random flip image.

    Args:
        flip_code: 0 means flipping vertically as also as flipping around the horizontal axis
            >= 1 means flipping horizontally as also as flipping around the vertical axis
            <= -1 means flipping around both axes
        probability: probability of random flip image

    Returns:
        True if successful, False otherwise.

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
        tensor_list_variable_axis=None,
        is_dynamic=False,
    ):
        if tensor_list_variable_axis is not None:
            assert isinstance(tensor_list_variable_axis, int)
        self._blobs.append(
            dict(
                name=name,
                data_source=data_source,
                shape=shape,
                dtype=dtype,
                tensor_list_variable_axis=tensor_list_variable_axis,
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
            if blob_conf.data_source == data_util.DataSourceCase.kImage:
                blob_conf.encode_case.jpeg.SetInParent()
            else:
                blob_conf.encode_case.raw.SetInParent()
            if blob["tensor_list_variable_axis"] is not None:
                blob_conf.tensor_list_variable_axis = blob["tensor_list_variable_axis"]
            blob_conf.is_dynamic = blob["is_dynamic"]
            op_conf.data_load_conf.blobs.extend([blob_conf])

            lbi = logical_blob_id_util.LogicalBlobId()
            lbi.op_name = op_conf.name
            lbi.blob_name = blob_conf.name
            self._outputs[blob_conf.name] = remote_blob_util.RemoteBlob(lbi)

        compile_context.CurJobAddConsistentOp(op_conf)


@oneflow_export("tensor_list_to_tensor_buffer")
def tensor_list_to_tensor_buffer(input, name=None):
    if name is None:
        name = id_util.UniqueStr("TensorListToBuffer_")
    assert isinstance(name, str)

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.tensor_list_to_tensor_buffer_conf, "in", input.logical_blob_name)
    setattr(op_conf.tensor_list_to_tensor_buffer_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("tensor_buffer_to_tensor_list")
def tensor_buffer_to_tensor_list(input, shape, dtype, name=None):
    if name is None:
        name = id_util.UniqueStr("TensorBufferToList_")

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.tensor_buffer_to_tensor_list_conf, "in", input.logical_blob_name)
    setattr(op_conf.tensor_buffer_to_tensor_list_conf, "out", "out")
    op_conf.tensor_buffer_to_tensor_list_conf.shape.dim[:] = list(shape)
    setattr(op_conf.tensor_buffer_to_tensor_list_conf, "data_type", dtype)
    compile_context.CurJobAddOp(op_conf)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("image_decode")
def image_decode(images_bytes_buffer, dtype=flow.uint8, color_space="BGR", name=None):
    # TODO: check color_space valiad
    if name is None:
        name = id_util.UniqueStr("ImageDecode_")

    op = (
        flow.user_op_builder(name)
        .Op("image_decode")
        .Input("in", [images_bytes_buffer])
        .Output("out")
        .Attr("color_space", color_space, "AttrTypeString")
        .Attr("data_type", dtype, "AttrTypeDataType")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("image_target_resize")
def image_target_resize(images, target_size, max_size, name=None):
    # TODO: check target_size and max_size valid
    if name is None:
        name = id_util.UniqueStr("ImageTargetResize_")

    op = (
        flow.user_op_builder(name)
        .Op("image_target_resize")
        .Input("in", [images])
        .Output("out")
        .Output("size")
        .Output("scale")
        .Attr("target_size", target_size, "AttrTypeInt32")
        .Attr("max_size", max_size, "AttrTypeInt32")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()


@oneflow_export("image_batch_align")
def image_batch_align(images, shape, dtype, alignment, name=None):
    if name is None:
        name = id_util.UniqueStr("ImageBatchAlign_")

    op = (
        flow.user_op_builder(name)
        .Op("image_batch_align")
        .Input("in", [images])
        .Output("out")
        .Attr("shape", shape, "AttrTypeShape")
        .Attr("data_type", dtype, "AttrTypeDataType")
        .Attr("alignment", alignment, "AttrTypeInt32")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("image_normalize")
def image_normalize(image, std, mean, name=None):
    if name is None:
        name = id_util.UniqueStr("ImageNormalize_")

    assert isinstance(std, (list, tuple))
    assert isinstance(mean, (list, tuple))

    op = (
        flow.user_op_builder(name)
        .Op("image_normalize")
        .Input("in", [image])
        .Output("out")
        .Attr("std", std, "AttrTypeListFloat")
        .Attr("mean", mean, "AttrTypeListFloat")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("image_flip")
def image_flip(image, flip_code, name=None):
    assert isinstance(image, BlobDef)

    if name is None:
        name = id_util.UniqueStr("ImageFlip_")

    if not isinstance(flip_code, BlobDef):
        assert isinstance(flip_code, int)
        flip_code = flow.constant(
            flip_code, shape=(image.shape[0],), dtype=flow.int8, name="{}_FlipCode_".format(name)
        )
    else:
        assert image.shape[0] == flip_code.shape[0]

    op = (
        flow.user_op_builder(name)
        .Op("image_flip")
        .Input("in", [image])
        .Input("flip_code", [flip_code])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("object_bbox_flip")
def object_bbox_flip(bbox, image_size, flip_code, name=None):
    assert isinstance(bbox, BlobDef)
    assert isinstance(image_size, BlobDef)
    assert bbox.shape[0] == image_size.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectBboxFlip_")

    if not isinstance(flip_code, BlobDef):
        assert isinstance(flip_code, int)
        flip_code = flow.constant(
            flip_code, shape=(bbox.shape[0],), dtype=flow.int8, name="{}_FlipCode".format(name)
        )
    else:
        assert bbox.shape[0] == flip_code.shape[0]

    op = (
        flow.user_op_builder(name)
        .Op("object_bbox_flip")
        .Input("bbox", [bbox])
        .Input("image_size", [image_size])
        .Input("flip_code", [flip_code])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("object_bbox_scale")
def object_bbox_scale(bbox, scale, name=None):
    assert isinstance(bbox, BlobDef)
    assert isinstance(scale, BlobDef)
    assert bbox.shape[0] == scale.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectBboxScale_")

    op = (
        flow.user_op_builder(name)
        .Op("object_bbox_scale")
        .Input("bbox", [bbox])
        .Input("scale", [scale])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("object_segmentation_polygon_flip")
def object_segm_poly_flip(poly, image_size, flip_code, name=None):
    assert isinstance(poly, BlobDef)
    assert isinstance(image_size, BlobDef)
    assert poly.shape[0] == image_size.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectSegmPolyFilp_")

    if not isinstance(flip_code, BlobDef):
        assert isinstance(flip_code, int)
        flip_code = flow.constant(
            flip_code, shape=(poly.shape[0],), dtype=flow.int8, name="{}_FlipCode".format(name)
        )
    else:
        assert poly.shape[0] == flip_code.shape[0]

    op = (
        flow.user_op_builder(name)
        .Op("object_segmentation_polygon_flip")
        .Input("poly", [poly])
        .Input("image_size", [image_size])
        .Input("flip_code", [flip_code])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("object_segmentation_polygon_scale")
def object_segm_poly_scale(poly, scale, name=None):
    assert isinstance(poly, BlobDef)
    assert isinstance(scale, BlobDef)
    assert poly.shape[0] == scale.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectSegmPolyFilp_")

    op = (
        flow.user_op_builder(name)
        .Op("object_segmentation_polygon_scale")
        .Input("poly", [poly])
        .Input("scale", [scale])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("object_segmentation_polygon_to_mask")
def object_segm_poly_to_mask(poly, poly_index, image_size, name=None):
    assert isinstance(poly, BlobDef)
    assert isinstance(poly_index, BlobDef)
    assert isinstance(image_size, BlobDef)
    assert poly.shape[0] == poly_index.shape[0]
    assert poly.shape[0] == image_size.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectSegmPolyToMask_")

    op = (
        flow.user_op_builder(name)
        .Op("object_segmentation_polygon_to_mask")
        .Input("poly", [poly])
        .Input("poly_index", [poly_index])
        .Input("image_size", [image_size])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()[0]
