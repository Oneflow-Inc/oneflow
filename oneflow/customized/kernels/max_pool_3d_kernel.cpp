#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

template<typename T>
class CPUMaxPool3DKernel final : public user_op::OpKernel {
 public:
  CPUMaxPool3DKernel() = default;
  ~CPUMaxPool3DKernel() = default;

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
    const int32_t dim = 3;
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
          params_3d->Get(), x, y, GetMinVal<T>,
          [](const T& lhs, T& rhs) {
            if (lhs > rhs) { rhs = lhs; }
          },
          [](const int64_t size, T& out) {});
    } else if (data_format == "channels_last") {
      PoolKernelUtil<T>::CLastForward(
          params_3d->Get(), x, y, GetMinVal<T>,
          [](const int64_t in_col, const int64_t out_col, ConstEigenMatrixMap<T>& in_mat,
             EigenMatrixMap<T>& out_mat) {
            out_mat.col(out_col) = out_mat.col(out_col).cwiseMax(in_mat.col(in_col));
          },
          [](const int64_t size, const int64_t col, EigenMatrixMap<T>& out_mat) {});
    } else {
      UNIMPLEMENTED();
    }
  };
};

#define REGISTER_CPU_MAX_POOL_3D_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("max_pool_3d")                                               \
      .SetCreateFn<CPUMaxPool3DKernel<dtype>>()                                     \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kCPU                                \
               && y_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_CPU_MAX_POOL_3D_KERNEL(float)
REGISTER_CPU_MAX_POOL_3D_KERNEL(double)

template<typename T>
class CpuMaxPool3DGradKernel final : public user_op::OpKernel {
 public:
  CpuMaxPool3DGradKernel() = default;
  ~CpuMaxPool3DGradKernel() = default;

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
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    // TODO: tsai: reset op kernel state when is_dynamic if ready
    const OpKernelStateWrapper<Params3D>* params_3d =
        dynamic_cast<OpKernelStateWrapper<Params3D>*>(state);
    CHECK(params_3d != nullptr);
    const std::string data_format = ctx->GetAttr<std::string>("data_format");
    if (data_format == "channels_first") {
      PoolKernelUtil<T>::CFirstBackward(
          params_3d->Get(), dy, y, x, dx,
          [](const T& in, const T& out, const T& out_diff, const int64_t size, T& in_diff) {
            if (in == out) { in_diff += out_diff; }
          });
    } else if (data_format == "channels_last") {
      PoolKernelUtil<T>::CLastBackward(
          params_3d->Get(), dy, y, x, dx,
          [](const int64_t out_col, const int64_t in_col, const int64_t size,
             ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
             ConstEigenArrayMap<T>& out_diff_arr, EigenArrayMap<T>& in_diff_arr) {
            in_diff_arr.col(in_col) +=
                out_diff_arr.col(out_col)
                * (in_arr.col(in_col).cwiseEqual(out_arr.col(out_col)).template cast<T>());
          });
    } else {
      UNIMPLEMENTED();
    }
  };
};

#define REGISTER_CPU_MAX_POOL_3D_GRAD_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("max_pool_3d_grad")                                            \
      .SetCreateFn<CpuMaxPool3DGradKernel<dtype>>()                                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kCPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_CPU_MAX_POOL_3D_GRAD_KERNEL(float)
REGISTER_CPU_MAX_POOL_3D_GRAD_KERNEL(double)

}  // namespace oneflow
