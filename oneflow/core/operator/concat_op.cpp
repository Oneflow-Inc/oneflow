#include "oneflow/core/operator/concat_op.h"

namespace oneflow {

void ConcatOp::InitFromOpConf() {
  CHECK(op_conf().has_concat_conf());

  for (int i = 0; i < op_conf().concat_conf().in_size(); ++i) {
    std::string ibn = "in_" + std::to_string(i);
    CHECK(ibn2lbn_.emplace(ibn, op_conf().concat_conf().in(i)).second);
    EnrollInputBn(ibn);
  }
  EnrollOutputBn("out");
}

const PbMessage& ConcatOp::GetSpecialConf() const {
  return op_conf().concat_conf();
}

void ConcatOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ConcatOpConf& conf = op_conf().concat_conf();
  const BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp(input_bns().at(0));
  std::vector<int64_t> out_dim_vec = in_0_blob_desc->shape().dim_vec();
  int32_t concat_axis = conf.axis();
  if (concat_axis < 0) { concat_axis += out_dim_vec.size(); }
  for (size_t i = 1; i < input_bns().size(); ++i) {
    const BlobDesc* in_i_blob_desc = GetBlobDesc4BnInOp(input_bns().at(i));
    for (int64_t j = 0; j < in_i_blob_desc->shape().NumAxes(); ++j) {
      if (j == concat_axis) {
        out_dim_vec[j] += in_i_blob_desc->shape().At(j);
      } else {
        CHECK_EQ(out_dim_vec[j], in_i_blob_desc->shape().At(j));
      }
    }
    CHECK_EQ(in_i_blob_desc->data_type(), in_0_blob_desc->data_type());
    CHECK_EQ(in_i_blob_desc->has_data_id(), in_0_blob_desc->has_data_id());
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(out_dim_vec);
  out_blob_desc->set_data_type(in_0_blob_desc->data_type());
  out_blob_desc->set_has_data_id(in_0_blob_desc->has_data_id());
}

void ConcatOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const BlobDesc* out_blob = GetBlobDesc4BnInOp(kernel_conf->output_bns(0));

  int32_t concat_axis = op_conf().concat_conf().axis();
  CHECK_NE(concat_axis, 0);

  if (concat_axis < 0) { concat_axis += out_blob->shape().NumAxes(); }
  int64_t dim_cp_num = 1;
  if (concat_axis != (out_blob->shape().NumAxes() - 1)) {
    dim_cp_num = out_blob->shape().Count(concat_axis + 1);
  }
  int64_t total_cp_num = 1;
  if (concat_axis != 0) {
    total_cp_num = out_blob->shape().Count(0, concat_axis);
  }

  std::vector<int64_t> per_cp_bytesize;
  for (const std::string& ibn : kernel_conf->input_bns()) {
    const BlobDesc* in_blob = GetBlobDesc4BnInOp(ibn);
    const int64_t in_concat_axis_dim = in_blob->shape().At(concat_axis);
    const int64_t cp_bytesize = in_concat_axis_dim * dim_cp_num
                                * GetSizeOfDataType(kernel_conf->data_type());
    per_cp_bytesize.push_back(cp_bytesize);
  }

  ConcatKernelConf* concat_kernel_conf = kernel_conf->mutable_concat_conf();
  concat_kernel_conf->set_total_cp_num(total_cp_num);
  *(concat_kernel_conf->mutable_per_cp_bytesize()) =
      StdVec2PbRf(per_cp_bytesize);
}

REGISTER_OP(OperatorConf::kConcatConf, ConcatOp);

}  // namespace oneflow
