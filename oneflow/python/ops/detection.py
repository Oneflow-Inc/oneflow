from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.lib.core.pb_util as pb_util

from oneflow.python.oneflow_export import oneflow_export
import oneflow as flow

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

@oneflow_export("detection.anchor_boxes_size")
class AnchorBoxesSize(object):
    def __init__(self, width, height):
        assert isinstance(width, int)
        assert isinstance(height, int)

        self.width = width
        self.height = height

    def to_proto(self, proto=None):
        proto = proto or op_conf_util.AnchorBoxesSize()
        proto.width = self.width
        proto.height = self.height
        return proto


@oneflow_export("detection.yolo_detect")
def yolo_detect(bbox, probs, image_height, image_width, image_origin_height, image_origin_width, layer_height, layer_width, prob_thresh, num_classes, anchor_boxes_size, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("YoloDetect_"),
    )
    setattr(op_conf.yolo_detect_conf, "bbox", bbox.logical_blob_name)
    setattr(op_conf.yolo_detect_conf, "probs", probs.logical_blob_name)
    op_conf.yolo_detect_conf.image_height = image_height
    op_conf.yolo_detect_conf.image_width = image_width
    op_conf.yolo_detect_conf.image_origin_height = image_origin_height
    op_conf.yolo_detect_conf.image_origin_width = image_origin_width
    op_conf.yolo_detect_conf.layer_height = layer_height
    op_conf.yolo_detect_conf.layer_width = layer_width
    op_conf.yolo_detect_conf.prob_thresh = prob_thresh
    op_conf.yolo_detect_conf.num_classes = num_classes
    assert isinstance(anchor_boxes_size, (list, tuple))
    op_conf.yolo_detect_conf.anchor_boxes_size.extend([anchor_box.to_proto() for anchor_box in anchor_boxes_size]) 
    op_conf.yolo_detect_conf.out_bbox = ("out_bbox")
    op_conf.yolo_detect_conf.out_probs = ("out_probs")
    op_conf.yolo_detect_conf.valid_num = ("valid_num")
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for bn in ["out_bbox", "out_probs", "valid_num"]:
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = bn
        ret.append(remote_blob_util.RemoteBlob(lbi))
    return ret


@oneflow_export("detection.yolo_box_diff")
def yolo_box_diff(bbox, gt_boxes, gt_labels, gt_valid_num, image_height, image_width, layer_height, layer_width, ignore_thresh, truth_thresh, box_mask, anchor_boxes_size, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("YoloBoxDiff_"),
    )
    setattr(op_conf.yolo_box_diff_conf, "bbox", bbox.logical_blob_name)
    setattr(op_conf.yolo_box_diff_conf, "gt_boxes", gt_boxes.logical_blob_name)
    setattr(op_conf.yolo_box_diff_conf, "gt_labels", gt_labels.logical_blob_name)
    setattr(op_conf.yolo_box_diff_conf, "gt_valid_num", gt_valid_num.logical_blob_name)
    op_conf.yolo_box_diff_conf.image_height = image_height
    op_conf.yolo_box_diff_conf.image_width = image_width
    op_conf.yolo_box_diff_conf.layer_height = layer_height
    op_conf.yolo_box_diff_conf.layer_width = layer_width
    op_conf.yolo_box_diff_conf.ignore_thresh = ignore_thresh
    op_conf.yolo_box_diff_conf.truth_thresh = truth_thresh
    assert isinstance(anchor_boxes_size, (list, tuple))
    op_conf.yolo_box_diff_conf.anchor_boxes_size.extend([anchor_box.to_proto() for anchor_box in anchor_boxes_size]) 
    op_conf.yolo_box_diff_conf.box_mask.extend([mask for mask in box_mask])
    op_conf.yolo_box_diff_conf.bbox_loc_diff = "bbox_loc_diff"
    op_conf.yolo_box_diff_conf.pos_inds = "pos_inds"
    op_conf.yolo_box_diff_conf.pos_cls_label = "pos_cls_label"
    op_conf.yolo_box_diff_conf.neg_inds = "neg_inds"
    op_conf.yolo_box_diff_conf.valid_num = "valid_num"
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for bn in ["bbox_loc_diff", "pos_inds", "pos_cls_label",  "neg_inds", "valid_num"]:
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = bn
        ret.append(remote_blob_util.RemoteBlob(lbi))
    return ret



@oneflow_export("detection.yolo_prob_loss")
def yolo_prob_loss(bbox_objness, bbox_clsprob, pos_inds, pos_cls_label, neg_inds, valid_num, num_classes, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("YoloBoxDiff_"),
    )
    setattr(op_conf.yolo_prob_loss_conf, "bbox_objness", bbox_objness.logical_blob_name)
    setattr(op_conf.yolo_prob_loss_conf, "bbox_clsprob", bbox_clsprob.logical_blob_name)
    setattr(op_conf.yolo_prob_loss_conf, "pos_inds", pos_inds.logical_blob_name)
    setattr(op_conf.yolo_prob_loss_conf, "pos_cls_label", pos_cls_label.logical_blob_name)
    setattr(op_conf.yolo_prob_loss_conf, "neg_inds", neg_inds.logical_blob_name)
    setattr(op_conf.yolo_prob_loss_conf, "valid_num", valid_num.logical_blob_name)
    op_conf.yolo_prob_loss_conf.num_classes = num_classes
    op_conf.yolo_prob_loss_conf.bbox_objness_out = "bbox_objness_out"
    op_conf.yolo_prob_loss_conf.bbox_clsprob_out = "bbox_clsprob_out"
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for bn in ["bbox_objness_out", "bbox_clsprob_out"]:
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = bn
        ret.append(remote_blob_util.RemoteBlob(lbi))
    return ret
