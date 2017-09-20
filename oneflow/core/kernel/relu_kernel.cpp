#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_data = BnInOp2Blob("in");
  if (in_data->has_data_id()) {
    CopyDataIdFromIbToAllOb<device_type>(ctx.device_ctx, BnInOp2Blob);
  }
  Blob* out_data = BnInOp2Blob("out");
  out_data->CopyDataIdFrom<device_type>(ctx.device_ctx, in_data);
  ReluKernelUtil<device_type, T>::Forward(ctx, out_data->shape().elem_cnt(),
                                          in_data->dptr<T>(),
                                          out_data->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_data = BnInOp2Blob("in");
  const Blob* out_diff = BnInOp2Blob("out_diff");
  Blob* in_diff = BnInOp2Blob("in_diff");
  in_diff->CopyDataIdFrom<device_type>(ctx.device_ctx, out_diff);
  ReluKernelUtil<device_type, T>::Backward(
      ctx, in_data->shape().elem_cnt(), out_diff->dptr<T>(), in_data->dptr<T>(),
      in_diff->mut_dptr<T>());
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
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
          RELU_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
          FLOATING_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ)};

  return creators.at(
      GetHashKey(op_ctx.device_type(), op_ctx.bn_in_op2data_type().at("in")))();
}

COMMAND(AddKernelCreator(OperatorConf::kReluConf, CreateReluKernel));

}  // namespace oneflow
