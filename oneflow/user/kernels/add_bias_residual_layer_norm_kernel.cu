#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include <cub/cub.cuh>
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/layer_norm.cuh"
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000

namespace oneflow {

namespace {

} // namespace

template<typename T>
class AddBiasResidualLayerNormGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  AddBiasResidualLayerNormGpuKernel() = default;
  ~AddBiasResidualLayerNormGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // obtain x and check its shape
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const ShapeView &x_shape = x->shape_view();
    CHECK_GT(x_shape.NumAxes(), 1)
      << "number of axes of \'x\' should have be greater than 1, yet get " << x_shape.NumAxes();

    // obtain bias and check its shape
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    const ShapeView &bias_shape = bias->shape_view();
    CHECK_EQ(bias_shape.NumAxes(), 1)
        << "number of axes of \'bias\' should have be greater than 1, yet get "
        << bias_shape.NumAxes();
    CHECK_EQ(bias_shape.At(0), x_shape.At(x_shape.NumAxes() - 1))
        << "dimension 1 of \'bias\'(" << bias_shape.At(0)
        << ") is not consistant with the last dimension of \'x\'("
        << x_shape.At(x_shape.NumAxes() - 1) << ")";
    
    // obtain gamma and check its shape
    const T* gamma_ptr = nullptr;
    ShapeView gamma_shape;
    if (ctx->has_input("gamma", 0)) {
        const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
        gamma_shape = gamma->shape_view();
        gamma_ptr = gamma->dptr<T>();
        CHECK_EQ(gamma_shape, bias_shape);
    }

    // obtain residual and check their shape
    const T* residual_1_ptr = nullptr;
    const T* residual_2_ptr = nullptr;
    ShapeView residual_1_shape;
    ShapeView residual_2_shape;
    if (ctx->has_input("residual_1", 0)) {
        const user_op::Tensor* residual_1 = ctx->Tensor4ArgNameAndIndex("residual_1", 0);
        residual_1_shape = residual_1->shape_view();
        residual_1_ptr = residual_1->dptr<T>();
        CHECK_EQ(residual_1_shape, x_shape);
    }
    if (ctx->has_input("residual_2", 0)) {
        CHECK(ctx->has_input("residual_1"))
            << "must provide residual_1 while residual_2 is provided";
        const user_op::Tensor* residual_2 = ctx->Tensor4ArgNameAndIndex("residual_2", 0);
        residual_2_shape = residual_2->shape_view();
        residual_2_ptr = residual_2->dptr<T>();
        CHECK_EQ(residual_2_shape, x_shape);
    }

    // obtain beta and check its shape
    const T* beta_ptr = nullptr;
    ShapeView beta_shape;
    if (ctx->has_input("beta", 0)) {
        const user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
        beta_shape = beta->shape_view();
        beta_ptr = beta->dptr<T>();
        CHECK_EQ(beta_shape, bias_shape);
    }

    // obtain output tensors
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const ShapeView &y_shape = y->shape_view();
    const ShapeView &mean_shape = mean->shape_view();
    const ShapeView &inv_variance_shape = inv_variance->shape_view();
    // TODO

    // obtain epsilon and check its value
    const double epsilon = ctx->Attr<double>("epsilon");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);


    
    // check shape
  }
}

} // namespace oneflow