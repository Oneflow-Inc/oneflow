#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
ReluKernel<device_type, T>::~ReluKernel() {
  if (!op_conf().use_cudnn_on_gpu()) { return; }
  CudaCheck(cudnnDestroyTensorDescriptor(in_desc_));
  CudaCheck(cudnnDestroyTensorDescriptor(out_desc_));
  CudaCheck(cudnnDestroyActicationDescriptor(activation_desc_));
}

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::VirtualKernelInit(const ParallelContext*) {
  if (!op_conf().use_cudnn_on_gpu()) { return; }
  CudaCheck(cudnnCreateTensorDescriptor(&in_desc_));
  CudaCheck(cudnnCreateTensorDescriptor(&out_desc_));
  CudaCheck(cudnnCreateActicationDescriptor(&activation_desc_));
  CudaCheck(cudnnActicationDescriptor(
      activation_desc_, CUDNN_ACTIVATION_RELU, CUDNN_PROPOGATE_NAN, 0.0));
}

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  KernelUtil<device_type, T>::Relu(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                   in_blob->dptr<T>(), out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_blob = BnInOp2Blob("out");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  KernelUtil<device_type, T>::ReluBackward(
      ctx.device_ctx, out_diff_blob->shape().elem_cnt(),
      out_diff_blob->dptr<T>(), out_blob->dptr<T>(),
      in_diff_blob->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReluConf, ReluKernel,
                           FLOATING_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ);

}  // namespace oneflow
