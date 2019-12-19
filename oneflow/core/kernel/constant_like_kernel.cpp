#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/constant_like_kernel_util.h"

namespace oneflow {

template<typename T>
struct ConstantLikeUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const T scalar, T* out_ptr) {
    FOR_RANGE(size_t, i, 0, elem_cnt) { out_ptr[i] = scalar; }
  }
};

template<DeviceType device_type, typename T>
class ConstantLikeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantLikeKernel);
  ConstantLikeKernel() = default;
  ~ConstantLikeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* out_blob = BnInOp2Blob("out");
    ConstantLikeUtil<device_type, T>::Forward(
        ctx.device_ctx, out_blob->shape().elem_cnt(),
        static_cast<T>(this->op_conf().constant_like_conf().scalar()), out_blob->mut_dptr<T>());
  }
};

#define REGISTER_CONSTANT_LIKE_KERNEL(dtype)                                                      \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kConstantLikeConf, DeviceType::kCPU, dtype, \
                                        ConstantLikeKernel<DeviceType::kCPU, dtype>)              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kConstantLikeConf, DeviceType::kGPU, dtype, \
                                        ConstantLikeKernel<DeviceType::kGPU, dtype>)

REGISTER_CONSTANT_LIKE_KERNEL(float);
REGISTER_CONSTANT_LIKE_KERNEL(double);
REGISTER_CONSTANT_LIKE_KERNEL(int8_t);
REGISTER_CONSTANT_LIKE_KERNEL(int32_t);
REGISTER_CONSTANT_LIKE_KERNEL(int64_t);

}  // namespace oneflow
