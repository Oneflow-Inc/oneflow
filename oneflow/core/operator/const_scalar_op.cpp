#include "oneflow/core/operator/const_scalar_op.h"

namespace oneflow {

void ConstScalarOp::InitFromOpConf() {
  CHECK(op_conf().has_const_scalar_conf());
  EnrollInputBn("tick", false);
  EnrollOutputBn("out", false);
}

const PbMessage& ConstScalarOp::GetCustomizedConf() const { return op_conf().const_scalar_conf(); }

void ConstScalarOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const ConstScalarOpConf& conf = op_conf().const_scalar_conf();
  const DataType& data_type =
      conf.has_data_type() ? conf.data_type() : Global<JobDesc>::Get()->DefaultDataType();
  if (IsIntegralDataType(data_type)) {
    CHECK(conf.has_int_val());
  } else if (IsFloatingDataType(data_type)) {
    CHECK(conf.has_float_val());
  } else {
    UNIMPLEMENTED();
  }
  CHECK_GE(conf.rank(), 1);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->set_data_type(data_type);
  out->mut_shape() = Shape(std::vector<int64_t>(conf.rank(), 1));
}

REGISTER_OP(OperatorConf::kConstScalarConf, ConstScalarOp);

}  // namespace oneflow
