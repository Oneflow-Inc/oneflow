from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.distribute as distribute_util

from oneflow.python.oneflow_export import oneflow_export
import oneflow as flow


@oneflow_export("detection.roi_align")
def roi_align(
    x,
    rois,
    pooled_h,
    pooled_w,
    name=None,
    spatial_scale=0.0625,
    sampling_ratio=2,
):
    assert isinstance(pooled_h, int)
    assert isinstance(pooled_w, int)
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("RoiAlign_"),
    )
    op_conf.roi_align_conf.x = x.logical_blob_name
    op_conf.roi_align_conf.rois = rois.logical_blob_name
    op_conf.roi_align_conf.y = "out"
    op_conf.roi_align_conf.roi_align_args.pooled_h = pooled_h
    op_conf.roi_align_conf.roi_align_args.pooled_w = pooled_w
    op_conf.roi_align_conf.roi_align_args.spatial_scale = float(spatial_scale)
    op_conf.roi_align_conf.roi_align_args.sampling_ratio = int(sampling_ratio)
    op_conf.roi_align_conf.roi_align_args.data_format = "channels_first"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export(
    "detection.maskrcnn_positive_negative_sample", "detection.pos_neg_sampler"
)
def pos_neg_sampler(
    pos_inds, neg_inds, total_subsample_num, pos_fraction, name=None
):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name
        if name is not None
        else id_util.UniqueStr("MaskrcnnPositiveNegativeSample_"),
    )
    op_conf.maskrcnn_positive_negative_sample_conf.pos_inds = (
        pos_inds.logical_blob_name
    )
    op_conf.maskrcnn_positive_negative_sample_conf.neg_inds = (
        neg_inds.logical_blob_name
    )
    op_conf.maskrcnn_positive_negative_sample_conf.total_subsample_num = (
        total_subsample_num
    )
    op_conf.maskrcnn_positive_negative_sample_conf.pos_fraction = pos_fraction
    op_conf.maskrcnn_positive_negative_sample_conf.sampled_pos_inds = (
        "sampled_pos_inds"
    )
    op_conf.maskrcnn_positive_negative_sample_conf.sampled_neg_inds = (
        "sampled_neg_inds"
    )
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for bn in ["sampled_pos_inds", "sampled_neg_inds"]:
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = bn
        ret.append(remote_blob_util.RemoteBlob(lbi))
    return ret


@oneflow_export("detection.calc_iou_matrix")
def calc_iou_matrix(boxes1, boxes2, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("CalcIoUMatrix_"),
    )
    op_conf.calc_iou_matrix_conf.boxes1 = boxes1.logical_blob_name
    op_conf.calc_iou_matrix_conf.boxes2 = boxes2.logical_blob_name
    op_conf.calc_iou_matrix_conf.iou_matrix = "iou_matrix"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "iou_matrix"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("detection.box_encode")
def box_encode(ref_boxes, boxes, regression_weights, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BoxEncode_"),
    )
    op_conf.box_encode_conf.ref_boxes = ref_boxes.logical_blob_name
    op_conf.box_encode_conf.boxes = boxes.logical_blob_name
    op_conf.box_encode_conf.boxes_delta = "boxes_delta"
    regression_weights_proto = op_conf_util.BBoxRegressionWeights()
    regression_weights_proto.weight_x = regression_weights["weight_x"]
    regression_weights_proto.weight_y = regression_weights["weight_y"]
    regression_weights_proto.weight_h = regression_weights["weight_h"]
    regression_weights_proto.weight_w = regression_weights["weight_w"]
    op_conf.box_encode_conf.regression_weights.CopyFrom(
        regression_weights_proto
    )
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "boxes_delta"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("detection.box_decode")
def box_decode(ref_boxes, boxes_delta, regression_weights, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("BoxDecode_"),
    )
    op_conf.box_decode_conf.ref_boxes = ref_boxes.logical_blob_name
    op_conf.box_decode_conf.boxes_delta = boxes_delta.logical_blob_name
    regression_weights_proto = op_conf_util.BBoxRegressionWeights()
    regression_weights_proto.weight_x = regression_weights["weight_x"]
    regression_weights_proto.weight_y = regression_weights["weight_y"]
    regression_weights_proto.weight_h = regression_weights["weight_h"]
    regression_weights_proto.weight_w = regression_weights["weight_w"]
    op_conf.box_decode_conf.regression_weights.CopyFrom(
        regression_weights_proto
    )
    op_conf.box_decode_conf.boxes = "boxes"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "boxes"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("detection.level_map")
def level_map(
    inputs,
    min_level=2,
    max_level=5,
    canonical_level=4,
    canonical_scale=224,
    epsilon=1e-6,
    name=None,
):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("LevelMap_"),
    )
    setattr(op_conf.level_map_conf, "in", inputs.logical_blob_name)
    op_conf.level_map_conf.min_level = min_level
    op_conf.level_map_conf.max_level = max_level
    op_conf.level_map_conf.canonical_level = canonical_level
    op_conf.level_map_conf.canonical_scale = canonical_scale
    op_conf.level_map_conf.epsilon = epsilon
    op_conf.level_map_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("detection.anchor_generate")
def anchor_generate(
    images, feature_map_stride, aspect_ratios, anchor_scales, name=None
):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("AnchorGenerate_"),
    )
    assert isinstance(aspect_ratios, (list, tuple))
    if isinstance(anchor_scales, (list, tuple)) == False:
        anchor_scales = [anchor_scales]
    assert isinstance(anchor_scales, (list, tuple))
    setattr(op_conf.anchor_generate_conf, "images", images.logical_blob_name)
    op_conf.anchor_generate_conf.feature_map_stride = feature_map_stride
    op_conf.anchor_generate_conf.aspect_ratios.extend(aspect_ratios)
    op_conf.anchor_generate_conf.anchor_scales.extend(anchor_scales)
    op_conf.anchor_generate_conf.anchors = "anchors"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "anchors"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("detection.identify_non_small_boxes")
def identify_non_small_boxes(inputs, min_size=0.0, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name
        if name is not None
        else id_util.UniqueStr("IdentifyNonSmallBoxes_"),
    )
    setattr(
        op_conf.identify_non_small_boxes_conf, "in", inputs.logical_blob_name
    )
    op_conf.identify_non_small_boxes_conf.min_size = min_size
    op_conf.identify_non_small_boxes_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("detection.identify_outside_anchors")
def identify_outside_anchors(anchors, image_size, tolerance=0.0, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name
        if name is not None
        else id_util.UniqueStr("IdentifyOutsideAnchors_"),
    )
    op_conf.identify_outside_anchors_conf.anchors = anchors.logical_blob_name
    op_conf.identify_outside_anchors_conf.image_size = (
        image_size.logical_blob_name
    )
    op_conf.identify_outside_anchors_conf.tolerance = tolerance
    op_conf.identify_outside_anchors_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("detection.clip_to_image")
def clip_boxes_to_image(boxes, image_size, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ClipBoxesToImage_"),
    )
    op_conf.clip_boxes_to_image_conf.boxes = boxes.logical_blob_name
    op_conf.clip_boxes_to_image_conf.image_size = image_size.logical_blob_name
    op_conf.clip_boxes_to_image_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("detection.extract_piece_slice_id")
def extract_piece_slice_id(inputs, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ExtractPieceSliceId_"),
    )
    getattr(op_conf.extract_piece_slice_id_conf, "in").extend(
        [i.logical_blob_name for i in inputs]
    )
    op_conf.extract_piece_slice_id_conf.out.extend(
        ["out_" + str(i) for i in range(len(inputs))]
    )
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for i in range(len(inputs)):
        out_lbi = logical_blob_id_util.LogicalBlobId()
        setattr(out_lbi, "op_name", op_conf.name)
        setattr(out_lbi, "blob_name", "out_" + str(i))
        ret.append(remote_blob_util.RemoteBlob(out_lbi))
    return tuple(ret)


@oneflow_export("detection.nms")
def non_maximum_suppression(
    inputs, nms_iou_threshold=0.7, post_nms_top_n=1000, name=None
):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name
        if name is not None
        else id_util.UniqueStr("NonMaximumSuppression_"),
    )
    setattr(
        op_conf.non_maximum_suppression_conf, "in", inputs.logical_blob_name
    )
    op_conf.non_maximum_suppression_conf.nms_iou_threshold = nms_iou_threshold
    op_conf.non_maximum_suppression_conf.post_nms_top_n = post_nms_top_n
    op_conf.non_maximum_suppression_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("detection.smooth_l1")
def smooth_l1(prediction, label, beta=1.0, scale=1.0, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("SmoothL1_"),
    )
    op_conf.smooth_l1_conf.prediction = prediction.logical_blob_name
    op_conf.smooth_l1_conf.label = label.logical_blob_name
    op_conf.smooth_l1_conf.beta = float(beta)
    op_conf.smooth_l1_conf.scale = float(scale)
    op_conf.smooth_l1_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("detection.upsample_nearest")
def upsample_nearest(inputs, scale, data_format, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("UpsampleNearest_"),
    )
    assert isinstance(scale, int)
    setattr(op_conf.upsample_nearest_conf, "in", inputs.logical_blob_name)
    op_conf.upsample_nearest_conf.scale = scale
    op_conf.upsample_nearest_conf.data_format = data_format
    op_conf.upsample_nearest_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("detection.affine_channel", "layers.affine_channel")
def affine_channel(
    inputs,
    axis,
    use_bias=True,
    scale_initializer=None,
    bias_initializer=None,
    activation=None,
    trainable=True,
    name=None,
    model_distribute=distribute_util.broadcast(),
):
    name_prefix = (
        name if name is not None else id_util.UniqueStr("AffineChannel_")
    )
    if axis < 0:
        axis = axis + len(inputs.shape)
    assert axis >= 0 and axis < len(inputs.shape)
    scale_shape = [1] * len(inputs.shape)
    scale_shape[axis] = inputs.shape[axis]
    scale = flow.get_variable(
        name="{}-scale".format(name_prefix),
        shape=scale_shape,
        dtype=inputs.dtype,
        initializer=(
            scale_initializer
            if scale_initializer is not None
            else flow.constant_initializer(0)
        ),
        trainable=trainable,
        model_name="scale",
        distribute=model_distribute,
    )
    scale = scale.with_distribute(model_distribute)
    out = inputs * scale
    if use_bias:
        bias = flow.get_variable(
            name="{}-bias".format(name_prefix),
            shape=(inputs.shape[axis],),
            dtype=inputs.dtype,
            initializer=(
                bias_initializer
                if bias_initializer is not None
                else flow.constant_initializer(0)
            ),
            trainable=trainable,
            model_name="bias",
            distribute=model_distribute,
        )
        bias = bias.with_distribute(model_distribute)
        out = flow.nn.bias_add(
            out, bias, name="{}_bias_add".format(name_prefix)
        )
    out = activation(out) if activation is not None else out
    return out


@oneflow_export("dim0_dynamic_to_fixed")
def dim0_dynamic_to_fixed(inputs, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("Dim0DynamicToFixed_"),
    )
    getattr(op_conf.dim0_dynamic_to_fixed_conf, "in").extend(
        [i.logical_blob_name for i in inputs]
    )
    op_conf.dim0_dynamic_to_fixed_conf.out.extend(
        ["out_" + str(i) for i in range(len(inputs))]
    )
    setattr(op_conf.dim0_dynamic_to_fixed_conf, "mask", "mask")
    compile_context.CurJobAddOp(op_conf)

    ret = []
    for i in range(len(inputs) + 1):
        out_lbi = logical_blob_id_util.LogicalBlobId()
        setattr(out_lbi, "op_name", op_conf.name)
        if i == len(inputs):
            setattr(out_lbi, "blob_name", "mask")
        else:
            setattr(out_lbi, "blob_name", "out_" + str(i))
        ret.append(remote_blob_util.RemoteBlob(out_lbi))
    return tuple(ret)


@oneflow_export("detection.maskrcnn_split")
def maskrcnn_split(input, segms, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("MaskrcnnSplit_"),
    )
    setattr(op_conf.maskrcnn_split_conf, "in", input.logical_blob_name)
    op_conf.maskrcnn_split_conf.segm[:] = [
        segm.logical_blob_name for segm in segms
    ]
    op_conf.maskrcnn_split_conf.out[:] = [
        "out_" + str(i) for i in range(len(segms))
    ]
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for i in range(len(segms)):
        out_lbi = logical_blob_id_util.LogicalBlobId()
        setattr(out_lbi, "op_name", op_conf.name)
        setattr(out_lbi, "blob_name", "out_" + str(i))
        ret.append(remote_blob_util.RemoteBlob(out_lbi))
    return ret


@oneflow_export("detection.masks_crop_and_resize")
def masks_crop_and_resize(masks, rois, mask_h, mask_w, name=None):
    assert len(masks.shape) == 4
    assert len(rois.shape) == 2
    assert masks.shape[0] == rois.shape[0]
    assert isinstance(mask_h, int)
    assert isinstance(mask_w, int)

    name = name or id_util.UniqueStr("masks_crop_and_resize_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    op_conf.masks_crop_and_resize_conf.masks = masks.logical_blob_name
    op_conf.masks_crop_and_resize_conf.rois = rois.logical_blob_name
    op_conf.masks_crop_and_resize_conf.mask_height = mask_h
    op_conf.masks_crop_and_resize_conf.mask_width = mask_w
    op_conf.masks_crop_and_resize_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
