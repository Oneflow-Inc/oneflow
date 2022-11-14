#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/include/primitive/unary_op.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/ep/cuda/primitive/unary_functor.cuh"
#include "oneflow/core/kernel/util/cuda_half_util.h"

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace {

template<typename T, ep::primitive::UnaryOp act_type>
__global__ void FusedGluForwardGpu(
    const int64_t m, const int64_t n, const int64_t k, 
    const int64_t stride,
    ep::primitive::UnaryFunctor<DeviceType::kCUDA, act_type, T, T>& act,
    const T* matmul_wx, const T* b,
    const T* matmul_vx, const T* c, T* y) {
  CUDA_1D_KERNEL_LOOP(i, m*n){
    // obtain the row and col index in output tensor "y"
    const int64_t y_row = i/n;
    const int64_t y_col = i - y_row*n;
    
    // calculate the hidden_state and gate
    T hidden_state = matmul_wx[stride*y_row+y_col] + b[y_col];
    T gate = matmul_vx[stride*y_row+y_col] + c[y_col];

    // calculate activation
    T act_gate = act(gate);

    // calculate element-wise product
    y[i] = hidden_state*act_gate;
  }
}


} // namespace


template<typename T>
void DispatchFusedGluForwardGpu(
    ep::Stream* stream,
    const int64_t m, const int64_t n, const int64_t k,
    int64_t stride,
    const T* matmul_wx, const T* b, 
    const T* matmul_vx, const T* c, 
    T* y,
    const std::string& activation){
  if(activation == "none"){
    ep::primitive::UnaryFunctor<DeviceType::kCUDA, ep::primitive::UnaryOp::kIdentity, T, T> act(0, 0);
    RUN_CUDA_KERNEL((FusedGluForwardGpu<T, ep::primitive::UnaryOp::kIdentity>),
      /* CUDA stream */ stream,
      /* number of threads */ m*n,
      /* args */ m, n, k, stride, act, matmul_wx, b, matmul_vx, c, y
    );
  } else if(activation == "sigmoid") {
    ep::primitive::UnaryFunctor<DeviceType::kCUDA, ep::primitive::UnaryOp::kSigmoid, T, T> act(0, 0);
    RUN_CUDA_KERNEL((FusedGluForwardGpu<T, ep::primitive::UnaryOp::kSigmoid>),
      /* CUDA stream */ stream,
      /* number of threads */ m*n,
      /* args */ m, n, k, stride, act, matmul_wx, b, matmul_vx, c, y
    );
  } else if(activation == "relu") {
    ep::primitive::UnaryFunctor<DeviceType::kCUDA, ep::primitive::UnaryOp::kRelu, T, T> act(0, 0);
    RUN_CUDA_KERNEL((FusedGluForwardGpu<T, ep::primitive::UnaryOp::kRelu>),
      /* CUDA stream */ stream,
      /* number of threads */ m*n,
      /* args */ m, n, k, stride, act, matmul_wx, b, matmul_vx, c, y
    );
  } else if(activation == "gelu") {
    ep::primitive::UnaryFunctor<DeviceType::kCUDA, ep::primitive::UnaryOp::kGelu, T, T> act(0, 0);
    RUN_CUDA_KERNEL((FusedGluForwardGpu<T, ep::primitive::UnaryOp::kGelu>),
      /* CUDA stream */ stream,
      /* number of threads */ m*n,
      /* args */ m, n, k, stride, act, matmul_wx, b, matmul_vx, c, y
    );
  } else if(activation == "fast_gelu") {
    ep::primitive::UnaryFunctor<DeviceType::kCUDA, ep::primitive::UnaryOp::kFastGelu, T, T> act(0, 0);
    RUN_CUDA_KERNEL((FusedGluForwardGpu<T, ep::primitive::UnaryOp::kFastGelu>),
      /* CUDA stream */ stream,
      /* number of threads */ m*n,
      /* args */ m, n, k, stride, act, matmul_wx, b, matmul_vx, c, y
    );
  } else if(activation == "silu") {
    ep::primitive::UnaryFunctor<DeviceType::kCUDA, ep::primitive::UnaryOp::kSilu, T, T> act(0, 0);
    RUN_CUDA_KERNEL((FusedGluForwardGpu<T, ep::primitive::UnaryOp::kSilu>),
      /* CUDA stream */ stream,
      /* number of threads */ m*n,
      /* args */ m, n, k, stride, act, matmul_wx, b, matmul_vx, c, y
    );
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
class GpuFusedGluKernel final : public user_op::OpKernel {
 public:
  GpuFusedGluKernel() = default;
  ~GpuFusedGluKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // obtain tensors from context
    const user_op::Tensor *input_tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor *input_tensor_w = ctx->Tensor4ArgNameAndIndex("w", 0);
    const user_op::Tensor *input_tensor_b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor *out_tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor *out_tensor_matmul_wx = ctx->Tensor4ArgNameAndIndex("matmul_wx", 0);

    // obtain optional tensors from context
    bool is_split_mode = false;
    user_op::Tensor *input_tensor_v;
    user_op::Tensor *input_tensor_c;
    user_op::Tensor *out_tensor_matmul_vx;

    if(ctx->has_input("v", 0) && ctx->has_input("c", 0)){
      input_tensor_v = ctx->Tensor4ArgNameAndIndex("v", 0);
      input_tensor_c = ctx->Tensor4ArgNameAndIndex("c", 0);
      out_tensor_matmul_vx = ctx->Tensor4ArgNameAndIndex("matmul_vx", 0);
      is_split_mode=true;
    }
    
    // TODO: validate dimension and number of axes
  
    // infer m, n, k
    const int64_t input_x_num_axes = input_tensor_x->shape_view().NumAxes();
    const int64_t m = input_tensor_x->shape_view().Count(0, input_x_num_axes-1);
    const int64_t n = out_tensor_y->shape_view().At(input_x_num_axes-1);
    const int64_t k = input_tensor_x->shape_view().At(input_x_num_axes-1);

    // calculate matmul_wx (and matmul_vx) through cuBLAS
    auto matmul = ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(
        DeviceType::kCUDA, input_tensor_x->data_type(), ep::primitive::BlasTransposeType::N,
        ep::primitive::BlasTransposeType::T);
    CHECK(matmul);
    /* Launch(Stream* stream, size_t m, size_t n, size_t k, Scalar alpha, const void* a,
                  const void* b, Scalar beta, void* c) = 0; */
    if (is_split_mode) {
      matmul->Launch(ctx->stream(), m, n, k, 1.0, input_tensor_x->dptr(), input_tensor_w->dptr(),
                      0.0, out_tensor_matmul_wx->mut_dptr());
      matmul->Launch(ctx->stream(), m, n, k, 1.0, input_tensor_x->dptr(), input_tensor_v->dptr(),
                      0.0, out_tensor_matmul_vx->mut_dptr());
    } else {
      matmul->Launch(ctx->stream(), m, n*2, k, 1.0, input_tensor_x->dptr(), input_tensor_w->dptr(),
                      0.0, out_tensor_matmul_wx->mut_dptr());
    }
    
    // dispatch according to activation type
    DispatchFusedGluForwardGpu<T>(
      ctx->stream(),
      /*m, n, k=*/m, n, k, 
      /*stride=*/ is_split_mode ? n : 2*n,
      /*matmul_wx=*/ out_tensor_matmul_wx->dptr<T>(),
      /*b=*/ input_tensor_b->dptr<T>(),
      /*matmul_vx=*/ is_split_mode ? out_tensor_matmul_vx->dptr<T>() : out_tensor_matmul_wx->dptr<T>()+n,
      /*c=*/ is_split_mode ? input_tensor_c->dptr<T>() : input_tensor_b->dptr<T>()+n,
      /*y=*/ out_tensor_y->mut_dptr<T>(),
      /*activation=*/ ctx->Attr<std::string>("activation")
    );
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_FUSED_GLU_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL("fused_glu")                                     \
      .SetCreateFn<GpuFusedGluKernel<dtype>>()                          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)  \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GPU_FUSED_GLU_KERNEL(float)
REGISTER_GPU_FUSED_GLU_KERNEL(double)
REGISTER_GPU_FUSED_GLU_KERNEL(half)
REGISTER_GPU_FUSED_GLU_KERNEL(nv_bfloat16)

} // namespace oneflow