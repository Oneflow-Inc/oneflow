#include "oneflow/core/kernel/concat_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  int32_t axis = this->op_conf().concat_conf().axis();
  if (axis < 0) { axis += out_blob->shape().NumAxes(); }
  const int64_t row_num = out_blob->shape().elem_cnt() / out_blob->shape().Count(axis);
  const int64_t out_col_num = out_blob->shape().Count(axis);
  int64_t out_col_offset = 0;
  for (const auto& input_bn : this->op_attribute().input_bns()) {
    const Blob* in_blob = BnInOp2Blob(input_bn);
    const int64_t in_col_num = in_blob->shape().Count(axis);
    CHECK_EQ(in_blob->shape().elem_cnt(), row_num * in_col_num);
    CHECK_EQ(in_blob->data_type(), out_blob->data_type());
    if (row_num * in_col_num > 0) {
      KernelUtil<device_type, T>::CopyColsRegion(
          ctx.device_ctx, row_num, in_col_num, in_blob->dptr<T>(), 0, in_col_num,
          out_blob->mut_dptr<T>(), out_col_offset, out_col_num);
      out_col_offset += in_col_num;
    }
  }
  CHECK_EQ(out_col_offset, out_col_num);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConcatConf, ConcatKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
