#include "oneflow/core/operator/bitonic_sort_op.h"

namespace oneflow {

namespace {

template<typename T>
const bool IsIntegerPowerOf2(const T v) {
  return (v > 0 && !(v & (v - 1)));
}

}  // namespace

void BitonicSortOp::InitFromOpConf() {
  CHECK(op_conf().has_bitonic_sort_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& BitonicSortOp::GetCustomizedConf() const {
  return this->op_conf().bitonic_sort_conf();
}

void BitonicSortOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  // input
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in->shape().NumAxes(), 2);
  CHECK(IsIntegerPowerOf2(in->shape().elem_cnt()));
  // output
  *GetBlobDesc4BnInOp("out") = *in;
}

REGISTER_OP(OperatorConf::kBitonicSortConf, BitonicSortOp);

}  // namespace oneflow
