#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
ReluKernel<device_type, T>::ReluKernel() {
#ifdef USE_CUDNN
  CudaCheck(cudnnCreateTensorDescriptor(&in_desc_));
  CudaCheck(cudnnCreateTensorDescriptor(&out_desc_));
  CudaCheck(cudnnCreateActivationDescriptor(&activ_desc_));
  CudaCheck(cudnnSetActivationDescriptor(activ_desc_, CUDNN_ACTIVATION_RELU,
                                         CUDNN_PROPAGATE_NAN, 0.0));
#endif  // USE_CUDNN
}

template<DeviceType device_type, typename T>
ReluKernel<device_type, T>::~ReluKernel() {
#ifdef USE_CUDNN
  CudaCheck(cudnnDestroyTensorDescriptor(in_desc_));
  CudaCheck(cudnnDestroyTensorDescriptor(out_desc_));
  CudaCheck(cudnnDestroyActivationDescriptor(activ_desc_));
#endif  // USE_CUDNN
}

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CopyDataIdFromSoleIbToAllObIfNeed<device_type>(ctx, BnInOp2Blob);

#ifdef USE_CUDNN
  CudaCheck(cudnnActivationForward(ctx.device_ctx->cudnn_handle(), activ_desc_,
                                   cudnn::DataType<T>::one, in_desc_,
                                   in_blob->dptr<T>(), cudnn::DataType<T>::zero,
                                   out_desc_, out_blob->mut_dptr<T>()));
#else
  ReluKernelUtil<device_type, T>::Forward(ctx, out_blob->shape().elem_cnt(),
                                          in_blob->dptr<T>(),
                                          out_blob->mut_dptr<T>());
#endif  // USE_CUDNN
}

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");

#ifdef USE_CUDNN
  const Blob* out_blob = BnInOp2Blob("out");
  CudaCheck(cudnnActivationBackward(
      ctx.device_ctx->cudnn_handle(), activ_desc_, cudnn::DataType<T>::one,
      out_desc_, out_blob->dptr<T>(), out_desc_, out_diff_blob->dptr<T>(),
      in_desc_, in_blob->dptr<T>(), cudnn::DataType<T>::zero, in_desc_,
      in_diff_blob->mut_dptr<T>()));
#else
  ReluKernelUtil<device_type, T>::Backward(
      ctx, in_blob->shape().elem_cnt(), out_diff_blob->dptr<T>(),
      in_blob->dptr<T>(), in_diff_blob->mut_dptr<T>());
#endif  // USE_CUDNN
}

template<typename T>
class ReluKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernelUtil);
  ReluKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t n, const T* in,
                      T* out) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        out[i] = std::max(in[i], static_cast<T>(0.0));
      }
    });
  }

  static void Backward(const KernelCtx& ctx, const int64_t n, const T* out_diff,
                       const T* in, T* in_diff) {
    ctx.device_ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < n; ++i) {
        in_diff[i] = in[i] > 0 ? out_diff[i] : 0;
      }
    });
  }
};

Kernel* CreateReluKernel(const OpContext& op_ctx) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define RELU_KERNEL_ENTRY(device_type, data_type_pair)                     \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)), []() {      \
     return new ReluKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>; \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(RELU_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ)};

  return creators.at(
      GetHashKey(op_ctx.device_type(), op_ctx.bn_in_op2data_type().at("in")))();
}

COMMAND(AddKernelCreator(OperatorConf::kReluConf, CreateReluKernel));

}  // namespace oneflow
