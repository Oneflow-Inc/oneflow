#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/include/primitive/unary_op.h"
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/ep/cuda/primitive/unary_functor.cuh"

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace {

template<typename T, typename IndexType, ep::primitive::UnaryOp act_type>
__global__ void FusedGluWithoutLinearGradGpu(
    const IndexType m, const IndexType n, const IndexType output_stride,
    ep::primitive::UnaryFunctor<DeviceType::kCUDA, act_type, T, T> act,
    const T* dy, const T* matmul_wx, const T* matmul_vx, 
    T* d_matmul_wx, T* d_matmul_vx
){
    
}

template<typename T>
class GpuFusedGluWithoutLinearGradKernel final : public user_op::OpKernel {
  public:
    GpuFusedGluWithoutLinearGradKernel() = default;
    ~GpuFusedGluWithoutLinearGradKernel() override = default;

  private:
    using user_op::OpKernel::Compute;
    void Compute(user_op::KernelComputeContext* ctx) const override {
        // obtain tensors from context
        const user_op::Tensor* input_tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
        const user_op::Tensor* input_tensor_matmul_wx = ctx->Tensor4ArgNameAndIndex("matmul_wx", 0);
        user_op::Tensor* out_tensor_d_matmul_wx = ctx->Tensor4ArgNameAndIndex("d_matmul_wx", 0);

        // obtain optional tensors from context
        bool is_split_mode = false;
        user_op::Tensor* input_tensor_matmul_vx = nullptr;
        user_op::Tensor* out_tensor_d_matmul_vx = nullptr;
        if (ctx->has_input("matmul_vx", 0)){
            input_tensor_matmul_vx = ctx->Tensor4ArgNameAndIndex("matmul_vx", 0);
            out_tensor_d_matmul_vx = ctx->Tensor4ArgNameAndIndex("d_matmul_vx", 0);
            is_split_mode = true;
        }

        // obtain tensor shapes and number of axes
        const ShapeView& dy_shape = input_tensor_dy->shape_view();
        const ShapeView& matmul_wx_shape = input_tensor_matmul_wx->shape_view();
        const ShapeView& d_matmul_wx_shape = out_tensor_d_matmul_wx->shape_view();
        const size_t dy_num_axes = dy_shape.NumAxes();
        const size_t matmul_wx_num_axes = matmul_wx_shape.NumAxes();

        // validate dimension and number of axes
        CHECK_GE_OR_RETURN(dy_num_axes, 2)
          << "number of axes of \'dy\' should have be greater than 1, yet get " << dy_num_axes;
        CHECK_GE_OR_RETURN(matmul_wx_num_axes, 2)
          << "number of axes of \'matmul_wx\' should have be greater than 1, yet get " << matmul_wx_num_axes;
        CHECK_EQ_OR_RETURN(dy_num_axes, matmul_wx_num_axes)
          << "number of axes of \'dy\'(" << dy_num_axes
          << ") is not consistant with the one of \'matmul_wx\'(" << matmul_wx_num_axes
          << ")";
        
        // check input shape
        if(is_split_mode){
            CHECK_EQ_OR_RETURN(2*dy_shape.At(dy_num_axes-1), matmul_wx_shape.At(matmul_wx_num_axes-1))
              << "two times of the last dimension of \'dy\'(" << 2*dy_shape.At(dy_num_axes-1)
              << ") is not consistant with the last dimension of \'matmul_wx\'(" 
              << matmul_wx_shape.At(matmul_wx_num_axes-1) << ")";
        } else {
            CHECK_EQ_OR_RETURN(dy_shape.At(dy_num_axes-1), matmul_wx_shape.At(matmul_wx_num_axes-1))
              << "the last dimension of \'dy\'(" << dy_shape.At(dy_num_axes-1)
              << ") is not consistant with the last dimension of \'matmul_wx\'(" 
              << matmul_wx_shape.At(matmul_wx_num_axes-1) << ")";
        }

        // check optional input tensor shapes
        if(is_split_mode){
            const Shape& matmul_vx_shape = ctx->InputShape("matmul_vx", 0);
            size_t matmul_vx_num_axes = matmul_vx_shape.NumAxes();
            CHECK_GE_OR_RETURN(matmul_vx_num_axes, 2)
              << "number of axes of \'matmul_vx\' should have be greater than 1, yet get " << matmul_vx_num_axes;
            CHECK_EQ_OR_RETURN(matmul_vx_num_axes, dy_num_axes)
              << "number of axes of \'dy\'(" << dy_num_axes
              << ") is not consistant with the one of \'matmul_vx\'(" << matmul_vx_num_axes
              << ")";
            CHECK_EQ_OR_RETURN(matmul_vx_shape.At(matmul_vx_num_axes-1), dy_shape.At(dy_num_axes-1))
              << "the last dimension of \'dy\'(" << dy_shape.At(dy_num_axes-1)
              << ") is not consistant with the last dimension of \'matmul_vx\'(" 
              << matmul_vx_shape.At(matmul_vx_num_axes-1) << ")";
        }

        // infer m, n
        const int64_t m = dy_shape.Count(0, dy_num_axes - 1);
        const int64_t n = dy_shape.At(dy_num_axes - 1);

        // todo: invoke kernel to process
    }

    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}

}