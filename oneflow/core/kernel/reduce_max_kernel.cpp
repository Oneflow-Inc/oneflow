#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ReduceMaxKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceMaxKernel);
  ReduceMaxKernel() = default;
  ~ReduceMaxKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    Blob* fw_tmp_blob = BnInOp2Blob("fw_tmp");
    const ReduceMaxOpConf& conf = this->op_conf().reduce_max_conf();
    const Shape& reduced_shape =
        conf.axis().empty()
            ? Shape::Ones(in_blob->shape().NumAxes())
            : in_blob->shape().CreateReducedShape({conf.axis().begin(), conf.axis().end()});
    NdarrayUtil<device_type, T>::ReduceMax(
        ctx.device_ctx, XpuVarNdarray<T>(reduced_shape, out_blob->mut_dptr<T>()),
        XpuVarNdarray<const T>(in_blob, in_blob->shape().NumAxes()),
        XpuVarNdarray<T>(fw_tmp_blob, in_blob->shape().NumAxes()));
  }
};

#define REGISTER_REDUCE_MAX_KERNEL(dtype)                                                      \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kReduceMaxConf, DeviceType::kGPU, dtype, \
                                        ReduceMaxKernel<DeviceType::kGPU, dtype>)              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kDeviceReduceMaxConf, DeviceType::kGPU,  \
                                        dtype, ReduceMaxKernel<DeviceType::kGPU, dtype>)       \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kReduceMaxConf, DeviceType::kCPU, dtype, \
                                        ReduceMaxKernel<DeviceType::kCPU, dtype>)              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kDeviceReduceMaxConf, DeviceType::kCPU,  \
                                        dtype, ReduceMaxKernel<DeviceType::kCPU, dtype>)

REGISTER_REDUCE_MAX_KERNEL(float);
REGISTER_REDUCE_MAX_KERNEL(double);
REGISTER_REDUCE_MAX_KERNEL(int8_t);
REGISTER_REDUCE_MAX_KERNEL(int32_t);
REGISTER_REDUCE_MAX_KERNEL(int64_t);

}  // namespace oneflow
