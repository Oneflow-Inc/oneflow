#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

#define DEFINE_BROADCAST_KERNEL_CLASS(type, abbr)                                                  \
  template<DeviceType device_type, typename T>                                                     \
  class Broadcast##type##Kernel final : public KernelIf<device_type> {                             \
   public:                                                                                         \
    OF_DISALLOW_COPY_AND_MOVE(Broadcast##type##Kernel);                                            \
    Broadcast##type##Kernel() = default;                                                           \
    ~Broadcast##type##Kernel() = default;                                                          \
                                                                                                   \
   private:                                                                                        \
    void ForwardDataContent(const KernelCtx& kernel_ctx,                                           \
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override { \
      const Blob* a_blob = BnInOp2Blob("a");                                                       \
      const Blob* b_blob = BnInOp2Blob("b");                                                       \
      Blob* out_blob = BnInOp2Blob("out");                                                         \
      size_t num_axes = out_blob->shape().NumAxes();                                               \
      NdarrayUtil<device_type, T>::Broadcast##abbr(                                                \
          kernel_ctx.device_ctx, XpuVarNdarray<int8_t>(out_blob, num_axes),                        \
          XpuVarNdarray<const T>(a_blob, num_axes), XpuVarNdarray<const T>(b_blob, num_axes));     \
    }                                                                                              \
  };                                                                                               \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcast##type##Conf, DeviceType::kGPU,    \
                                        int8_t,                                                    \
                                        Broadcast##type##Kernel<DeviceType::kGPU, int8_t>);        \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcast##type##Conf, DeviceType::kCPU,    \
                                        int8_t,                                                    \
                                        Broadcast##type##Kernel<DeviceType::kGPU, int8_t>);        \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcast##type##Conf, DeviceType::kGPU,    \
                                        int32_t,                                                   \
                                        Broadcast##type##Kernel<DeviceType::kGPU, int32_t>);       \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcast##type##Conf, DeviceType::kCPU,    \
                                        int32_t,                                                   \
                                        Broadcast##type##Kernel<DeviceType::kGPU, int32_t>);       \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcast##type##Conf, DeviceType::kGPU,    \
                                        int64_t,                                                   \
                                        Broadcast##type##Kernel<DeviceType::kGPU, int64_t>);       \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcast##type##Conf, DeviceType::kCPU,    \
                                        int64_t,                                                   \
                                        Broadcast##type##Kernel<DeviceType::kGPU, int64_t>);       \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcast##type##Conf, DeviceType::kGPU,    \
                                        float, Broadcast##type##Kernel<DeviceType::kGPU, float>);  \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcast##type##Conf, DeviceType::kGPU,    \
                                        double,                                                    \
                                        Broadcast##type##Kernel<DeviceType::kGPU, double>);        \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcast##type##Conf, DeviceType::kCPU,    \
                                        float, Broadcast##type##Kernel<DeviceType::kCPU, float>);  \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBroadcast##type##Conf, DeviceType::kCPU,    \
                                        double,                                                    \
                                        Broadcast##type##Kernel<DeviceType::kCPU, double>);

DEFINE_BROADCAST_KERNEL_CLASS(Equal, EQ)
DEFINE_BROADCAST_KERNEL_CLASS(NotEqual, NE);
DEFINE_BROADCAST_KERNEL_CLASS(GreaterThan, GT);
DEFINE_BROADCAST_KERNEL_CLASS(GreaterEqual, GE);
DEFINE_BROADCAST_KERNEL_CLASS(LessThan, LT);
DEFINE_BROADCAST_KERNEL_CLASS(LessEqual, LE);
DEFINE_BROADCAST_KERNEL_CLASS(LogicalAnd, AND);
}  // namespace oneflow
