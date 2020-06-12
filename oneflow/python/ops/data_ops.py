from __future__ import absolute_import

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.record.image_pb2 as image_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.framework.remote_blob import BlobDef
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("data.ImagePreprocessor")
class ImagePreprocessor(object):
    def __init__(self, preprocessor):
        assert isinstance(preprocessor, str)
        if preprocessor.lower() != "bgr2rgb" and preprocessor.lower() != "mirror":
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

        proto.jpeg.preprocess.extend([p.to_proto() for p in self.image_preprocessors])
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
        self, mean_values, std_values=(1.0, 1.0, 1.0), data_format="channels_last",
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
    op_conf.decode_ofrecord_conf.part_name_suffix_length = part_name_suffix_length
    if shuffle is True:
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
        op_conf.record_load_conf.part_name_suffix_length = part_name_suffix_length
    if shuffle:
        op_conf.record_load_conf.random_shuffle_conf.buffer_size = shuffle_buffer_size
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = name
    lbi.blob_name = "out"

    compile_context.CurJobAddConsistentOp(op_conf)
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("data.ofrecord_reader")
def ofrecord_reader(
    ofrecord_dir,
    batch_size=1,
    data_part_num=1,
    part_name_prefix="part-",
    part_name_suffix_length=-1,
    random_shuffle=False,
    shuffle_buffer_size=1024,
    shuffle_after_epoch=False,
    name=None,
):
    if name is None:
        name = id_util.UniqueStr("OFRecord_Reader_")

    return (
        flow.user_op_builder(name)
        .Op("OFRecordReader")
        .Output("out")
        .Attr("data_dir", ofrecord_dir, "AttrTypeString")
        .Attr("data_part_num", data_part_num, "AttrTypeInt32")
        .Attr("batch_size", batch_size, "AttrTypeInt32")
        .Attr("part_name_prefix", part_name_prefix, "AttrTypeString")
        .Attr("random_shuffle", random_shuffle, "AttrTypeBool")
        .Attr("shuffle_buffer_size", shuffle_buffer_size, "AttrTypeInt32")
        .Attr("shuffle_after_epoch", shuffle_after_epoch, "AttrTypeBool")
        .Attr("part_name_suffix_length", part_name_suffix_length, "AttrTypeInt32")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


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
            flow.random_uniform_initializer()
        )

    if tick:
        op_conf.decode_random_conf.tick = tick.unique_name
    op_conf.decode_random_conf.out = "out"

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"

    compile_context.CurJobAddConsistentOp(op_conf)
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("tensor_list_to_tensor_buffer")
def tensor_list_to_tensor_buffer(input, name=None):
    if name is None:
        name = id_util.UniqueStr("TensorListToBuffer_")
    assert isinstance(name, str)

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.tensor_list_to_tensor_buffer_conf, "in", input.unique_name)
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
    setattr(op_conf.tensor_buffer_to_tensor_list_conf, "in", input.unique_name)
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
            flip_code,
            shape=(image.shape[0],),
            dtype=flow.int8,
            name="{}_FlipCode_".format(name),
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
            flip_code,
            shape=(bbox.shape[0],),
            dtype=flow.int8,
            name="{}_FlipCode".format(name),
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
            flip_code,
            shape=(poly.shape[0],),
            dtype=flow.int8,
            name="{}_FlipCode".format(name),
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


@oneflow_export("data.coco_reader")
def api_coco_reader(
    annotation_file,
    image_dir,
    batch_size,
    shuffle=True,
    random_seed=None,
    group_by_aspect_ratio=True,
    stride_partition=True,
    name=None,
):
    import random
    import sys

    if name is None:
        name = id_util.UniqueStr("COCOReader_")

    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)

    op = (
        flow.user_op_builder(name)
        .Op("COCOReader")
        .Output("image")
        .Output("image_id")
        .Output("image_size")
        .Output("gt_bbox")
        .Output("gt_label")
        .Output("gt_segm")
        .Output("gt_segm_index")
        .Attr("annotation_file", annotation_file, "AttrTypeString")
        .Attr("image_dir", image_dir, "AttrTypeString")
        .Attr("batch_size", batch_size, "AttrTypeInt64")
        .Attr("shuffle_after_epoch", shuffle, "AttrTypeBool")
        .Attr("random_seed", random_seed, "AttrTypeInt64")
        .Attr("group_by_ratio", group_by_aspect_ratio, "AttrTypeBool")
        .Attr("stride_partition", stride_partition, "AttrTypeBool")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()
