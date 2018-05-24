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
  ForwardField(ctx, BnInOp2Blob, [](Blob* blob) { return blob->mut_data_id(); },
               [](Blob* blob) { return blob->data_id(); },
               [](Blob* blob) { return blob->ByteSizeOfDataIdField(); });
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ForwardColNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardField(ctx, BnInOp2Blob,
               [](Blob* blob) { return reinterpret_cast<char*>(blob->mut_col_num()); },
               [](Blob* blob) { return reinterpret_cast<const char*>(blob->col_num()); },
               [](Blob* blob) { return blob->ByteSizeOfColNumField(); });
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ForwardField(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    std::function<char*(Blob*)> GetOutBlobField, std::function<const char*(Blob*)> GetInBlobField,
    std::function<size_t(Blob*)> GetFieldSize) const {
  CHECK_GE(this->op_conf().concat_conf().axis(), 1);
  Blob* out_blob = BnInOp2Blob("out");
  Blob* in_0_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(0));
  size_t out_blob_field_size = GetFieldSize(out_blob);
  CHECK_EQ(out_blob_field_size, GetFieldSize(in_0_blob));
  Memcpy<device_type>(ctx.device_ctx, GetOutBlobField(out_blob), GetInBlobField(in_0_blob),
                      out_blob_field_size);
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
