#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/customized/kernels/radix_sort.cuh"

namespace oneflow {

template<typename T>
class GpuSortKernel final : public user_op::OpKernel {
 public:
  GpuSortKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  GpuSortKernel() = default;
  ~GpuSortKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    Memcpy<DeviceType::kGPU>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(),
                             in->shape().elem_cnt() * sizeof(T));
    const int32_t instance_size = in->shape().At(in->shape().NumAxes() - 1);
    const int32_t instance_num = in->shape().elem_cnt() / instance_size;
    const std::string& direction = ctx->GetAttr<std::string>("direction");
    if (direction == "ASCENDING") {
      SortKeysAscending(in->dptr<T>(), instance_num, instance_size, tmp_buffer->mut_dptr<void>(),
                        tmp_buffer->shape().elem_cnt(), out->mut_dptr<T>(),
                        ctx->device_ctx()->cuda_stream());
    } else if (direction == "DESCENDING") {
      SortKeysDescending(in->dptr<T>(), instance_num, instance_size, tmp_buffer->mut_dptr<void>(),
                         tmp_buffer->shape().elem_cnt(), out->mut_dptr<T>(),
                         ctx->device_ctx()->cuda_stream());
    } else {
      UNIMPLEMENTED();
    }
  };
};

#define REGISTER_GPU_SORT_KERNEL(dtype)                                                     \
  REGISTER_USER_KERNEL("sort")                                                              \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                     \
        return new GpuSortKernel<dtype>(ctx);                                               \
      })                                                                                    \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                 \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);     \
        return ctx.device_type() == DeviceType::kGPU                                        \
               && out_desc->data_type() == GetDataType<dtype>::value;                       \
      })                                                                                    \
      .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {                          \
        const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);                        \
        const int32_t instance_size = in_shape->dim_vec().back();                           \
        const int32_t instance_num = in_shape->elem_cnt() / instance_size;                  \
        const std::string& direction = ctx->GetAttr<std::string>("direction");              \
        if (direction == "ASCENDING") {                                                     \
          return InferTempStorageForSortKeysAscending<dtype>(instance_num, instance_size);  \
        } else if (direction == "DESCENDING") {                                             \
          return InferTempStorageForSortKeysDescending<dtype>(instance_num, instance_size); \
        } else {                                                                            \
          UNIMPLEMENTED();                                                                  \
        }                                                                                   \
      });

REGISTER_GPU_SORT_KERNEL(float)
REGISTER_GPU_SORT_KERNEL(double)
REGISTER_GPU_SORT_KERNEL(int32_t)
REGISTER_GPU_SORT_KERNEL(int64_t)

}  // namespace oneflow
