#include "oneflow/core/operator/reduce_sum_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void ReduceSumOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_sum_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (op_conf().reduce_sum_conf().has_in_sys()) {
    EnrollDataTmpBn("fw_tmp");
  } else {
    EnrollFwBufBn("fw_tmp");
  }
}

const PbMessage& ReduceSumOp::GetCustomizedConf() const { return op_conf().reduce_sum_conf(); }

void ReduceSumOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext*) const {
  const ReduceSumOpConf& conf = op_conf().reduce_sum_conf();
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  out_blob->set_data_type(in_blob->data_type());
  if (conf.keep_dims()) {
    out_blob->mut_shape() =
        in_blob->shape().CreateReducedShape({conf.axis().begin(), conf.axis().end()});
  } else {
    out_blob->mut_shape() =
        in_blob->shape().CreateReducedShape7DropDims({conf.axis().begin(), conf.axis().end()});
  }
}

REGISTER_OP(OperatorConf::kReduceSumConf, ReduceSumOp);

}  // namespace oneflow
