#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* in_data = BnInOp2BlobPtr("in");
  Blob* out_data = BnInOp2BlobPtr("out");
  ReluKernelUtil<device_type, T>::Forward(ctx, out_data->shape().elem_cnt(),
                                          in_data->dptr<T>(),
                                          out_data->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* in_data = BnInOp2BlobPtr("in");
  const Blob* out_diff = BnInOp2BlobPtr("out_diff");
  Blob* in_diff = BnInOp2BlobPtr("in_diff");
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

namespace {

template<DeviceType device_type>
Kernel* CreateReluKernel(const OperatorConf& op_conf) {
  static const HashMap<int, std::function<Kernel*()>> data_type2creator = {
#define RELU_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto, []() { return new ReluKernel<device_type, type_cpp>; }},
      OF_PP_FOR_EACH_TUPLE(RELU_KERNEL_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)};
  return data_type2creator.at(op_conf.relu_conf().in().data_type())();
}

}  // namespace

REGISTER_TEMPLATE_KERNEL_CREATOR(OperatorConf::kReluConf, CreateReluKernel);

}  // namespace oneflow
