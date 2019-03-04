#include "oneflow/core/operator/element_count_op.h"

namespace oneflow {

void ElementCountOp::InitFromOpConf() {
  CHECK(op_conf().has_element_count_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& ElementCountOp::GetCustomizedConf() const { return op_conf().element_count_conf(); }

void ElementCountOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  const ElementCountOpConf& conf = op_conf().element_count_conf();
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape({1});
  const DataType data_type = conf.has_data_type() ? conf.data_type() : Global<JobDesc>::Get()->DefaultDataType();
  out->set_data_type(data_type);
}

REGISTER_OP(OperatorConf::kElementCountConf, ElementCountOp);

}  // namespace oneflow
