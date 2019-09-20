from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export('detection.roi_align')
def roi_align(x, rois, pooled_h, pooled_w, name=None, spatial_scale=0.0625, sampling_ratio=2):
    assert isinstance(pooled_h, int)
    assert isinstance(pooled_w, int)
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name",
            name if name is not None else id_util.UniqueStr('RoiAlign_'))
    op_conf.roi_align_conf.x = x.logical_blob_name
    op_conf.roi_align_conf.rois = rois.logical_blob_name
    op_conf.roi_align_conf.y = "out"
    op_conf.roi_align_conf.roi_align_conf.pooled_h = pooled_h
    op_conf.roi_align_conf.roi_align_conf.pooled_w = pooled_w
    op_conf.roi_align_conf.roi_align_conf.spatial_scale = float(spatial_scale)
    op_conf.roi_align_conf.roi_align_conf.sampling_ratio = int(
        sampling_ratio)
    op_conf.roi_align_conf.roi_align_conf.data_format = "channels_first"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
