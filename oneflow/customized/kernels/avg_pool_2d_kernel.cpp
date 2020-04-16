#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

template<typename T>
class CPUAvgpool2dKernel final : public user_op::OpKernel {
 public:
  CPUAvgpool2dKernel() = default;
  ~CPUAvgpool2dKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
    const int32_t dim = 2;
    const std::string data_format = ctx->GetAttr<std::string>("data_format");
    const std::string padding = ctx->GetAttr<std::string>("padding");
    const std::vector<int32_t>& pool_size = ctx->GetAttr<std::vector<int32_t>>("pool_size");
    const std::vector<int32_t>& strides = ctx->GetAttr<std::vector<int32_t>>("strides");
    return std::make_shared<OpKernelStateWrapper<Params3D>>(dim, x_shape, data_format, padding,
                                                            pool_size, strides);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    // TODO: tsai: reset op kernel state when is_dynamic if ready
    const OpKernelStateWrapper<Params3D>* params_3d =
        dynamic_cast<OpKernelStateWrapper<Params3D>*>(state);
    CHECK(params_3d != nullptr);
    const std::string data_format = ctx->GetAttr<std::string>("data_format");
    if (data_format == "channels_first") {
      PoolKernelUtil<T>::CFirstForward(
          params_3d->Get(), x, y, GetZeroVal<T>, [](const T& lhs, T& rhs) { rhs += lhs; },
          [](const int64_t size, T& out) { out /= size; });
    } else if (data_format == "channels_last") {
      // PoolKernelUtil<T>::CLastForward(
      //     params_3d->Get(), x, y, GetZeroVal<T>,
      //     [](const int64_t in_col, const int64_t out_col, ConstEigenMatrixMap<T>& in_mat,
      //        EigenMatrixMap<T>& out_mat) { out_mat.col(out_col) += in_mat.col(in_col); },
      //     [](const int64_t size, const int64_t col, EigenMatrixMap<T>& out_mat) {
      //       out_mat.col(col) /= size;
      //     });
    } else {
      UNIMPLEMENTED();
    }
  };
};

#define REGISTER_CPU_AVG_POOL_2D_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("avg_pool_2d")                                               \
      .SetCreateFn<CPUAvgpool2dKernel<dtype>>()                                     \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kCPU                                \
               && y_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_CPU_AVG_POOL_2D_KERNEL(float)
REGISTER_CPU_AVG_POOL_2D_KERNEL(double)

template<typename T>
class CpuAvgpool2dGradKernel final : public user_op::OpKernel {
 public:
  CpuAvgpool2dGradKernel() = default;
  ~CpuAvgpool2dGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override{
      // add your code
  };
};

#define REGISTER_CPU_AVG_POOL_2D_GRAD_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("avg_pool_2d_grad")                                            \
      .SetCreateFn<CpuAvgpool2dGradKernel<dtype>>()                                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kCPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_CPU_AVG_POOL_2D_GRAD_KERNEL(float)
REGISTER_CPU_AVG_POOL_2D_GRAD_KERNEL(double)

}  // namespace oneflow
