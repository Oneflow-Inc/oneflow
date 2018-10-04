#include "oneflow/core/operator/fpn_distribute_op.h"

namespace oneflow {

void FpnDistributeOp::InitFromOpConf() {
  CHECK(op_conf().has_fpn_distribute_conf());
  EnrollInputBn("collected_rois", false);
  EnrollDataTmpBn("target_levels");
  EnrollRepeatedOutputBn("rois", false);
  EnrollOutputBn("roi_indices", false);
}

const PbMessage& FpnDistributeOp::GetCustomizedConf() const {
  return op_conf().fpn_distribute_conf();
}

void FpnDistributeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // in: (post_nms_topn, 5)
  const BlobDesc* collected_rois_blob_desc = GetBlobDesc4BnInOp("collected_rois");
  CHECK_EQ(collected_rois_blob_desc->shape().At(1), 5);
  CHECK_EQ(collected_rois_blob_desc->shape().NumAxes(), 2);
  // outs: (post_nms_topn, 5)
  FOR_RANGE(int32_t, idx, 0, RepeatedObnSize("rois")) {
    // TODO: change later when dynamic features are in place
    *GetBlobDesc4BnInOp(RepeatedObn("rois", idx)) = *collected_rois_blob_desc;
  }
  CHECK_EQ(RepeatedObnSize("rois"), op_conf().fpn_distribute_conf().roi_max_level()
                                        - op_conf().fpn_distribute_conf().roi_min_level() + 1);
  CHECK_GE(op_conf().fpn_distribute_conf().roi_max_level(),
           op_conf().fpn_distribute_conf().roi_canonical_level());
  CHECK_LE(op_conf().fpn_distribute_conf().roi_min_level(),
           op_conf().fpn_distribute_conf().roi_canonical_level());
  // roi_indices: (post_nms_topn)
  BlobDesc* roi_indices_blob_desc = GetBlobDesc4BnInOp("roi_indices");
  roi_indices_blob_desc->mut_shape() = Shape({collected_rois_blob_desc->shape().At(0)});
  roi_indices_blob_desc->set_data_type(DataType::kInt32);
  BlobDesc* target_levels_blob_desc = GetBlobDesc4BnInOp("target_levels");
  target_levels_blob_desc->mut_shape() = Shape({collected_rois_blob_desc->shape().At(0)});
  target_levels_blob_desc->set_data_type(DataType::kInt32);
}

REGISTER_OP(OperatorConf::kFpnDistributeConf, FpnDistributeOp);

}  // namespace oneflow
