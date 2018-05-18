#include "oneflow/core/kernel/concat_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int32_t axis = this->op_conf().concat_conf().axis();
  Blob* out_blob = BnInOp2Blob("out");
  int64_t row_num = out_blob->shape().elem_cnt() / out_blob->shape().Count(axis);
  int64_t output_col_num = out_blob->shape().Count(axis);
  int64_t output_col_offset = 0;
  for (const auto& input_bn : this->op_attribute().input_bns()) {
    const Blob* in_blob = BnInOp2Blob(input_bn);
    int64_t input_col_num = in_blob->shape().Count(axis);
    CHECK_EQ(in_blob->shape().elem_cnt(), row_num * input_col_num);
    CHECK_EQ(in_blob->data_type(), out_blob->data_type());
    KernelUtil<device_type, T>::CopyColsRegion(
        ctx.device_ctx, row_num, input_col_num, in_blob->dptr<T>(), 0, input_col_num,
        out_blob->mut_dptr<T>(), output_col_offset, output_col_num);
    output_col_offset += input_col_num;
  }
  CHECK_EQ(output_col_offset, output_col_num);
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  DataIdIterator input_it(BnInOp2Blob, &this->op_attribute().input_bns(),
                          this->op_conf().concat_conf().axis());
  DataIdIterator output_it(BnInOp2Blob, &this->op_attribute().output_bns(), 0);
  CopyFromIterToIter<device_type>(ctx.device_ctx, input_it, output_it);
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::ForwardColNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ColNumIterator input_it(BnInOp2Blob, &this->op_attribute().input_bns(),
                          this->op_conf().concat_conf().axis());
  ColNumIterator output_it(BnInOp2Blob, &this->op_attribute().output_bns(), 0);
  CopyFromIterToIter<device_type>(ctx.device_ctx, input_it, output_it);
}

template<DeviceType device_type, typename T>
void ConcatKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  DataContentIterator input_it(BnInOp2Blob, &this->op_attribute().output_diff_bns(), 0);
  DataContentIterator output_it(BnInOp2Blob, &this->op_attribute().input_diff_bns(),
                                this->op_conf().concat_conf().axis());
  CopyFromIterToIter<device_type>(ctx.device_ctx, input_it, output_it);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConcatConf, ConcatKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
