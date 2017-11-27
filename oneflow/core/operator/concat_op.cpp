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
  bool is_forward = kernel_conf->is_forward();
  const std::string& concat_out_bn =
      is_forward ? kernel_conf->output_bns(0) : kernel_conf->output_diff_bns(0);
  const BlobDesc* out_blob = GetBlobDesc4BnInOp(concat_out_bn);

  const int64_t& concat_axis = op_conf().concat_conf().axis();
  int64_t concat_element_cnt = 1;
  if ((concat_axis != (out_blob->shape().NumAxes() - 1))
      && (concat_axis != -1)) {
    concat_element_cnt = out_blob->shape().Count(concat_axis + 1);
  }
  int64_t concat_num_each_blob = 1;
  if ((concat_axis != (-out_blob->shape().NumAxes())) && (concat_axis != 0)) {
    concat_num_each_blob = out_blob->shape().Count(0, concat_axis);
  }

  std::vector<int64_t> cp_szs;
  const PbRpf<std::string>& concat_in_bns =
      is_forward ? kernel_conf->input_bns() : kernel_conf->input_diff_bns();
  for (const std::string& ibn : concat_in_bns) {
    const BlobDesc* in_blob = GetBlobDesc4BnInOp(ibn);
    const int64_t in_concat_axis_dim = in_blob->shape().At(concat_axis);
    const int64_t cp_sz = in_concat_axis_dim * concat_element_cnt
                          * GetSizeOfDataType(kernel_conf->data_type());
    cp_szs.push_back(cp_sz);
  }

  ConcatKernelConf* concat_kernel_conf = kernel_conf->mutable_concat_conf();
  if (is_forward) {
    concat_kernel_conf->set_fw_concat_element_cnt(concat_element_cnt);
    concat_kernel_conf->set_fw_concat_num_each_blob(concat_num_each_blob);
    *(concat_kernel_conf->mutable_fw_cp_szs()) = StdVec2PbRf(cp_szs);
  } else {
    concat_kernel_conf->set_bw_concat_element_cnt(concat_element_cnt);
    concat_kernel_conf->set_bw_concat_num_each_blob(concat_num_each_blob);
    *(concat_kernel_conf->mutable_bw_cp_szs()) = StdVec2PbRf(cp_szs);
  }
}

REGISTER_OP(OperatorConf::kConcatConf, ConcatOp);

}  // namespace oneflow
