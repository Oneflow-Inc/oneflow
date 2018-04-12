#include "oneflow/core/operator/decode_ofrecord_op.h"

namespace oneflow {

void DecodeOFRecordOp::InitFromOpConf() {
  CHECK(op_conf().has_decode_ofrecord_conf());
  const DecodeOFRecordOpConf& conf = op_conf().decode_ofrecord_conf();
  for (int32_t i = 0; i < conf.blob_size(); ++i) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
  if (conf.part_name_suffix_length() != -1) {
    CHECK_GE(
        conf.part_name_suffix_length(),
        std::to_string(Global<JobDesc>::Get()->job_conf().data_part_num() - 1)
            .length());
  }
}

void DecodeOFRecordOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->mutable_decode_ofrecord_conf()->set_random_seed(NewRandomSeed());
}

const PbMessage& DecodeOFRecordOp::GetCustomizedConf() const {
  return op_conf().decode_ofrecord_conf();
}

void DecodeOFRecordOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  FOR_RANGE(size_t, i, 0, output_bns().size()) {
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(output_bns().at(i));
    const BlobConf& blob_conf = op_conf().decode_ofrecord_conf().blob(i);
    std::vector<int64_t> dim_vec(1 + blob_conf.shape().dim_size());
    dim_vec[0] = Global<JobDesc>::Get()->SinglePieceSize();
    FOR_RANGE(size_t, j, 1, dim_vec.size()) {
      dim_vec[j] = blob_conf.shape().dim(j - 1);
    }
    out_blob_desc->mut_shape() = Shape(dim_vec);
    out_blob_desc->set_data_type(blob_conf.data_type());
    out_blob_desc->set_has_data_id_field(
        Global<JobDesc>::Get()->SizeOfOneDataId() > 0);
    out_blob_desc->set_has_col_num_field(blob_conf.max_sequence_size() > 1);
    out_blob_desc->set_max_col_num(blob_conf.max_sequence_size());
  }
}

std::string DecodeOFRecordOp::obn2lbn(const std::string& output_bn) const {
  CHECK(output_bn.substr(0, 4) == "out_");
  return op_name() + "/"
         + op_conf()
               .decode_ofrecord_conf()
               .blob(oneflow_cast<int32_t>(output_bn.substr(4)))
               .name();
}

REGISTER_OP(OperatorConf::kDecodeOfrecordConf, DecodeOFRecordOp);

}  // namespace oneflow
