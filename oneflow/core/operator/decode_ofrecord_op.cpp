#include "oneflow/core/operator/decode_ofrecord_op.h"

namespace oneflow {

void DecodeOfrecordOp::InitFromOpConf() {
  CHECK(op_conf().has_decode_ofrecord_conf());
  EnrollInputBn("in", false);
  const DecodeOfrecordOpConf& conf = op_conf().decode_ofrecord_conf();
  for(int32_t i = 0; i < conf.blobs_size(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
}

const PbMessage& DecodeOfrecordOp::GetCustomizedConf() const {
  return op_conf().decode_ofrecord_conf();
}

void DecodeOfrecordOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  
}

std::string DecodeOfrecordOp::obn2lbn(const std::string& output_bn) const {
  CHECK(output_bn.substr(0, 4) == "out_");
  return op_name() + "/" + op_conf().decode_ofrecord_conf().blobs(std::stoi(output_bn.substr(4))).name();
}


REGISTER_OP(OperatorConf::kDecodeOfrecordConf, DecodeOfrecordOp);

}  // namespace oneflow
