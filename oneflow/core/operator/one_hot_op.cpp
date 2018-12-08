#include "oneflow/core/operator/one_hot_op.h"

namespace oneflow {

void OneHotOp::InitFromOpConf() {
  CHECK(op_conf().has_one_hot_conf());
  EnrollInputBn("indices", false);
  EnrollOutputBn("out", false);
}

const PbMessage& OneHotOp::GetCustomizedConf() const { return op_conf().one_hot_conf(); }

void OneHotOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const OneHotOpConf& conf = op_conf().one_hot_conf();
  const int64_t depth = conf.depth();
  const DataType data_type =
      conf.has_data_type() ? conf.data_type() : Global<JobDesc>::Get()->DefaultDataType();
  CHECK_GT(depth, 0);
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIntegralDataType(indices->data_type()));
  CHECK_GT(indices->shape().NumAxes(), 0);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *indices;
  out->set_data_type(data_type);
  std::vector<int64_t> dim_vec = indices->shape().dim_vec();
  dim_vec.push_back(depth);
  out->mut_shape() = Shape(dim_vec);
}

REGISTER_OP(OperatorConf::kOneHotConf, OneHotOp);

}  // namespace oneflow
