#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename T>
class CpuInTopKKernel final : public user_op::OpKernel {
 public:
  CpuInTopKKernel() = default;
  ~CpuInTopKKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* predictions = ctx->Tensor4ArgNameAndIndex("predictions", 0);
    const user_op::Tensor* targets = ctx->Tensor4ArgNameAndIndex("targets", 0);

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    CHECK_EQ(predictions->shape().At(0), targets->shape().At(0));

    const int32_t targets_num = predictions->shape().At(0);
    const int32_t classes_num = predictions->shape().At(1);

    const T* prediction_ptr = predictions->dptr<T>();
    const int32_t* target_ptr = targets->dptr<int32_t>();
    const int32_t k = ctx->GetAttr<int32_t>("k");
    int8_t* out_ptr = out->mut_dptr<int8_t>();
    FOR_RANGE(int32_t, batch_idx, 0, targets_num) {
      int32_t target = target_ptr[batch_idx];

      bool cannot_say = (target >= classes_num)
                        || !std::isfinite(prediction_ptr[batch_idx * classes_num + target]);

      int32_t more_probable_classes = 0;
      if (!cannot_say) {
        const T target_prediction = prediction_ptr[batch_idx * classes_num + target];
        FOR_RANGE(int32_t, class_idx, 0, classes_num) {
          T pred = prediction_ptr[batch_idx * classes_num + class_idx];

          if (!std::isfinite(pred)) {
            cannot_say = true;
            break;
          } else if (pred > target_prediction) {
            ++more_probable_classes;
            if (more_probable_classes > k) break;
          }
        }
      }
      out_ptr[batch_idx] = cannot_say ? false : (more_probable_classes < k);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_IN_TOP_K_KERNEL(dtype)                                                       \
  REGISTER_USER_KERNEL("in_top_k")                                                                \
      .SetCreateFn<CpuInTopKKernel<dtype>>()                                                      \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                \
        const user_op::TensorDesc* pre_desc = ctx.TensorDesc4ArgNameAndIndex("predictions", 0);   \
        const user_op::TensorDesc* tar_desc = ctx.TensorDesc4ArgNameAndIndex("targets", 0);       \
        return ctx.device_type() == DeviceType::kCPU && tar_desc->data_type() == DataType::kInt32 \
               && pre_desc->data_type() == GetDataType<dtype>::value;                             \
      });

REGISTER_CPU_IN_TOP_K_KERNEL(float)
REGISTER_CPU_IN_TOP_K_KERNEL(double)

}  // namespace oneflow
