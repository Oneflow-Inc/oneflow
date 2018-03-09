#include "oneflow/core/operator/decode_ofrecord_op.h"

namespace oneflow {

void DecodeOfrecordOp::InitFromOpConf() {
  CHECK(op_conf().has_decode_ofrecord_conf());
  EnrollInputBn("in", false);
  for(int32_t i = 0; i < op_conf().decode_ofrecord_conf().blobs_size(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
}

const PbMessage& DecodeOfrecordOp::GetCustomizedConf() const {
  return op_conf().decode_ofrecord_conf();
}

void DecodeOfrecordOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const DecodeOfrecordOpConf& conf = op_conf().decode_ofrecord_conf();
  FOR_RANGE(size_t, i, 0, output_bns().size()) {
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(output_bns().at(i));
    std::vector<int64_t> dim_vec(1 + conf.blobs(i).shape().dim_size());
    dim_vec[0] = JobDesc::Singleton()->SinglePieceSize();
    FOR_RANGE(size_t, j, 1, dim_vec.size()) {
      dim_vec[j] = conf.blobs(i).shape().dim(j - 1);
    }
    out_blob_desc->mut_shape() = Shape(dim_vec);
    out_blob_desc->set_data_type(conf.blobs(i).data_type());
  }
}

std::string DecodeOfrecordOp::obn2lbn(const std::string& output_bn) const {
  CHECK(output_bn.substr(0, 4) == "out_");
  return op_name() + "/" + op_conf().decode_ofrecord_conf().blobs(std::stoi(output_bn.substr(4))).name();
}


REGISTER_OP(OperatorConf::kDecodeOfrecordConf, DecodeOfrecordOp);

}  // namespace oneflow
