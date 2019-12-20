from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.distribute as distribute_util

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
    assert isinstance(boxes_size, (list, tuple))
    for i in range(len(anchor_boxes_size)):
        #to confirm
        op_conf.yolo_detect_conf.add_anchor_boxes_size(width=anchor_boxes_size[i][0], height= anchor_boxes_size[i][1])

    op_conf.yolo_detect_conf.out_bbox = "out_bbox"
    op_conf.yolo_detect_conf.out_probs = "out_probs"
    op_conf.yolo_detect_conf.valid_num = "valid_num"
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for bn in ["out_bbox", "out_probs", "valid_num"]:
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = bn
        ret.append(remote_blob_util.RemoteBlob(lbi))
    return ret