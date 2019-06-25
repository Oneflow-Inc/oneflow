#include "oneflow/core/operator/local_nonzero_op.h"

namespace oneflow {

void LocalNonzeroOp::InitFromOpConf() {
  CHECK(op_conf().has_local_nonzero_conf());
  EnrollInputBn("in", false);
  if (this->device_type() == DeviceType::kGPU) {
    EnrollDataTmpBn("shape");
    EnrollDataTmpBn("num_nonzero");
  }
  EnrollOutputBn("out", false);
}

const PbMessage& LocalNonzeroOp::GetCustomizedConf() const {
  return this->op_conf().local_nonzero_conf();
}

void LocalNonzeroOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  // input
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const int64_t elem_cnt = in->shape().elem_cnt();
  const int64_t shape_dim = in->shape().NumAxes();
  if (this->device_type() == DeviceType::kGPU) {
    // data tmp: shape
    BlobDesc* shape = GetBlobDesc4BnInOp("shape");
    shape->mut_shape() = Shape({shape_dim});
    shape->set_data_type(DataType::kInt64);
    // data tmp: num_nonzero
    BlobDesc* num_nonzero = GetBlobDesc4BnInOp("num_nonzero");
    num_nonzero->mut_shape() = Shape({1});
    num_nonzero->set_data_type(DataType::kInt64);
  }
  // output
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape({elem_cnt, shape_dim});
  out->set_data_type(DataType::kInt32);
  out->set_has_dim0_valid_num_field(true);
  out->mut_dim0_inner_shape() = Shape({1, elem_cnt});
}

REGISTER_OP(OperatorConf::kLocalNonzeroConf, LocalNonzeroOp);

}  // namespace oneflow
