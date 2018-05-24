#include "oneflow/core/kernel/concat_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

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
void ConcatKernel<device_type, T>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardBlobField(ctx, BnInOp2Blob, &Blob::ByteSizeOfDataIdField, &Blob::CopyDataIdFrom);
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ForwardColNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardBlobField(ctx, BnInOp2Blob, &Blob::ByteSizeOfColNumField, &Blob::CopyColNumFrom);
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ForwardBlobField(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    SizeOfBlobFieldMthd SizeOf, CopyBlobFieldMthd Copy) const {
  CHECK_GE(this->op_conf().concat_conf().axis(), 1);
  Blob* out_blob = BnInOp2Blob("out");
  const Blob* in_0_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(0));
  size_t out_blob_field_size = (out_blob->*SizeOf)();
  CHECK_EQ(out_blob_field_size, (in_0_blob->*SizeOf)());
  (out_blob->*Copy)(ctx.device_ctx, in_0_blob);
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
