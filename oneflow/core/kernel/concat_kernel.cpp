#include "oneflow/core/kernel/concat_kernel.h"
#include "oneflow/core/kernel/iter_util.h"

namespace oneflow {

template<DeviceType device_type>
void ConcatKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  DataContentIterator input_it(BnInOp2Blob, &this->kernel_conf().input_bns(),
                               this->op_conf().concat_conf().axis());
  DataContentIterator output_it(BnInOp2Blob, &this->kernel_conf().output_bns(),
                                0);
  CopyFromIterToIter<DataContentIterator, device_type>(ctx.device_ctx, input_it,
                                                       output_it);
}

template<DeviceType device_type>
void ConcatKernel<device_type>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  DataIdIterator input_it(BnInOp2Blob, &this->kernel_conf().input_bns(),
                          this->op_conf().concat_conf().axis());
  DataIdIterator output_it(BnInOp2Blob, &this->kernel_conf().output_bns(), 0);
  CopyFromIterToIter<DataIdIterator, device_type>(ctx.device_ctx, input_it,
                                                  output_it);
}

template<DeviceType device_type>
void ConcatKernel<device_type>::ForwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ColNumIterator input_it(BnInOp2Blob, &this->kernel_conf().input_bns(),
                          this->op_conf().concat_conf().axis());
  ColNumIterator output_it(BnInOp2Blob, &this->kernel_conf().output_bns(), 0);
  CopyFromIterToIter<ColNumIterator, device_type>(ctx.device_ctx, input_it,
                                                  output_it);
}

template<DeviceType device_type>
void ConcatKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  DataContentIterator input_it(BnInOp2Blob,
                               &this->kernel_conf().output_diff_bns(), 0);
  DataContentIterator output_it(BnInOp2Blob,
                                &this->kernel_conf().input_diff_bns(),
                                this->op_conf().concat_conf().axis());
  CopyFromIterToIter<DataContentIterator, device_type>(ctx.device_ctx, input_it,
                                                       output_it);
}

namespace {

Kernel* CreateConcatKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define CONCAT_KERNEL_ENTRY(device_type) \
  {GetHashKey(device_type), []() { return new ConcatKernel<device_type>; }},
      OF_PP_FOR_EACH_TUPLE(CONCAT_KERNEL_ENTRY, DEVICE_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.device_type()))();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kConcatConf, CreateConcatKernel));

}  // namespace oneflow
