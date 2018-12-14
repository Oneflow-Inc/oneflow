#include "oneflow/core/kernel/concat_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int32_t axis = this->op_conf().concat_conf().axis();
  const PbRpf<std::string>& in_bns = this->op_attribute().input_bns();
  const Blob* in_0_blob = BnInOp2Blob(in_bns.Get(0));
  std::vector<int64_t> out_dim_vec = in_0_blob->shape().dim_vec();
  if (axis < 0) { axis += out_dim_vec.size(); }
  for (size_t i = 1; i < in_bns.size(); ++i) {
    const Blob* in_i_blob = BnInOp2Blob(in_bns.Get(i));
    for (int64_t j = 0; j < in_i_blob->shape().NumAxes(); ++j) {
      if (j == axis) {
        out_dim_vec[j] += in_i_blob->shape().At(j);
      } else {
        CHECK_EQ(out_dim_vec[j], in_i_blob->shape().At(j));
      }
    }
  }
  BnInOp2Blob("out")->set_instance_shape(Shape(out_dim_vec));
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int32_t axis = this->op_conf().concat_conf().axis();
  Blob* out_blob = BnInOp2Blob("out");
  const int64_t row_num = out_blob->shape().elem_cnt() / out_blob->shape().Count(axis);
  const int64_t out_col_num = out_blob->shape().Count(axis);
  int64_t out_col_offset = 0;
  for (const auto& input_bn : this->op_attribute().input_bns()) {
    const Blob* in_blob = BnInOp2Blob(input_bn);
    const int64_t in_col_num = in_blob->shape().Count(axis);
    CHECK_EQ(in_blob->shape().elem_cnt(), row_num * in_col_num);
    CHECK_EQ(in_blob->data_type(), out_blob->data_type());
    KernelUtil<device_type, T>::CopyColsRegion(
        ctx.device_ctx, row_num, in_col_num, in_blob->dptr<T>(), 0, in_col_num,
        out_blob->mut_dptr<T>(), out_col_offset, out_col_num);
    out_col_offset += in_col_num;
  }
  CHECK_EQ(out_col_offset, out_col_num);
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int32_t axis = this->op_conf().concat_conf().axis();
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const int64_t row_num = out_diff_blob->shape().elem_cnt() / out_diff_blob->shape().Count(axis);
  const int64_t out_diff_col_num = out_diff_blob->shape().Count(axis);
  int64_t out_diff_col_offset = 0;
  for (const auto& input_diff_bn : this->op_attribute().input_diff_bns()) {
    Blob* in_diff_blob = BnInOp2Blob(input_diff_bn);
    const int64_t in_diff_col_num = in_diff_blob->shape().Count(axis);
    CHECK_EQ(in_diff_blob->shape().elem_cnt(), row_num * in_diff_col_num);
    CHECK_EQ(in_diff_blob->data_type(), out_diff_blob->data_type());
    KernelUtil<device_type, T>::CopyColsRegion(
        ctx.device_ctx, row_num, in_diff_col_num, out_diff_blob->dptr<T>(), out_diff_col_offset,
        out_diff_col_num, in_diff_blob->mut_dptr<T>(), 0, in_diff_col_num);
    out_diff_col_offset += in_diff_col_num;
  }
  CHECK_EQ(out_diff_col_offset, out_diff_col_num);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConcatConf, ConcatKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
