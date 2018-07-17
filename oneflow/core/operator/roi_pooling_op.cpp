#include "oneflow/core/operator/roi_pooling_op.h"

namespace oneflow {

void RoIPoolingOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollInputBn("rois");
  EnrollOutputBn("out");
  EnrollDataTmpBn("argmax");
}

const PbMessage& RoIPoolingOp::GetCustomizedConf() const { return op_conf().roi_pooling_conf(); }

void RoIPoolingOp::InferBlobDescs(std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  // TODO check shape of in (feat. map)
  // TODO check shape of rois
  // TODO set shape of out, [batch num, roi num, channel num, pooled_h, pooled_w, c]
  // TODO set shape of argmax
}

REGISTER_OP(OperatorConf::kRoiPoolingConf, RoIPoolingOp);

}  // namespace oneflow
