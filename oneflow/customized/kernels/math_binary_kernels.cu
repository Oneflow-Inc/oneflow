#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include <math.h>

namespace oneflow {

namespace user_op {

#ifdef WITH_CUDA

typedef struct {
	float dx;
	float dy;
}stBinary;

__device__ void PowCallInDiff4GpuFloat(float x, float y, float dz, float& dx, float& dy)
{
	dx = dz * y * (powf(x, y-1));
	dy = dz * logf(x) * (powf(x, y));
}

// __device__ stBinary PowCallInDiff4GpuFloat(float x, float y, float dz)
// {
// 	stBinary m_stBry;
// 	m_stBry.dx = dz * y * (powf(x, y-1));
// 	m_stBry.dy = dz * logf(x) * (powf(x, y));
// 	return m_stBry;
// }

#define MATH_BINARY_GPU(func_name, fw_func, bw_func, dtype)                                                                          \
  __global__ void func_name##ForwardGpu(const int n, const dtype* x, const dtype* y, dtype* z){	                                     \
      CUDA_1D_KERNEL_LOOP(i, n) { z[i] = fw_func(x[i], y[i]); }                                                                      \
  }                                                                                                                                  \
  void func_name##Forward(DeviceCtx* ctx, const Tensor* tensor_x, const Tensor* tensor_y, Tensor* tensor_z){                         \
	  const dtype* x = tensor_x->dptr<dtype>();                                                                                      \
	  const dtype* y = tensor_y->dptr<dtype>();                                                                                      \
	  dtype* z = tensor_z->mut_dptr<dtype>();                                                                                        \
	  int64_t n = tensor_x->shape().elem_cnt();                                                                                      \
	  CHECK_LE(n, GetMaxVal<int32_t>() / 2);                                                                                         \
	  func_name##ForwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y, z);                \
  }                                                                                                                                  \
  __global__ void func_name##BackwardGpu(const int n, const dtype* x, const dtype* y, const dtype* dz, dtype* dx, dtype* dy){        \
	   CUDA_1D_KERNEL_LOOP(i, n){    																								 \
	       bw_func(x[i], y[i], dz[i], dx[i], dy[i]);                                                                                 \
	   }                                                                                                                             \
  }                                                                                                                                  \
  void func_name##Backward(DeviceCtx* ctx, const Tensor* tensor_x, const Tensor* tensor_y, const Tensor* tensor_dz,                  \
                          Tensor* tensor_dx, Tensor* tensor_dy) {                                                                    \
	  const dtype* x = tensor_x->dptr<dtype>();                                                                                      \
	  const dtype* y = tensor_y->dptr<dtype>();                                                                                      \
	  const dtype* dz = tensor_dz->dptr<dtype>();                                                                                    \
	  dtype* dx = tensor_dx->mut_dptr<dtype>();                                                                                      \
	  dtype* dy = tensor_dy->mut_dptr<dtype>();                                                                                      \
	  int64_t n = tensor_x->shape().elem_cnt();                                                                                      \
	  CHECK_LE(n, GetMaxVal<int32_t>() / 2);                                                                                         \
	  func_name##BackwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y, dz, dx, dy);      \
  }
	  
     

#define MATH_BINARY_GPU_FLOAT_SEQ    \
   OF_PP_MAKE_TUPLE_SEQ("Pow", Pow)



MATH_BINARY_GPU(Pow, powf, PowCallInDiff4GpuFloat, float);


class MathBinaryGpuFloatKernel final : public OpKernel
{
public:
	MathBinaryGpuFloatKernel(const KernelInitContext& ctx) : OpKernel(ctx) {}
	MathBinaryGpuFloatKernel() = default;
	~MathBinaryGpuFloatKernel() = default;
	
private:
	void Compute(KernelContext* ctx) override
	{
		const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
		const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
		Tensor* tensor_z = ctx->Tensor4ArgNameAndIndex("z", 0);
		std::string binary_math_type = ctx->GetAttr<std::string>("binary_math_type");
		
#define MATH_BINARY_FORWARD(binary_math_type_str, func_name_prefix)                \
 if(binary_math_type == binary_math_type_str){                                     \
	 func_name_prefix##Forward(ctx->device_ctx(), tensor_x, tensor_y, tensor_z);   \
 }																				   
 
  OF_PP_FOR_EACH_TUPLE(MATH_BINARY_FORWARD, MATH_BINARY_GPU_FLOAT_SEQ);
 #undef MATH_BINARY_FORWARD
	}
};

REGISTER_USER_KERNEL("binary")
    .SetCreateFn([](const KernelInitContext& ctx) { return new MathBinaryGpuFloatKernel(ctx); })
	.SetIsMatchedPred([](const KernelRegContext& ctx){
		if(ctx.device() == DeviceType::kGPU && ctx.data_type() == DataType::kFloat) { return true; }
		return false;
	});
	
class MathBinaryGradGpuFloatKernel final : public OpKernel 
{
public:
	MathBinaryGradGpuFloatKernel(const KernelInitContext& ctx) : OpKernel(ctx) {}
	MathBinaryGradGpuFloatKernel() = default;
	~MathBinaryGradGpuFloatKernel() = default;
	
private:
    void Compute(KernelContext* ctx) override 
    {
        const Tensor* tensor_x = ctx->Tensor4ArgNameAndIndex("x", 0);
        const Tensor* tensor_y = ctx->Tensor4ArgNameAndIndex("y", 0);
        const Tensor* tensor_dz = ctx->Tensor4ArgNameAndIndex("dz", 0);
        Tensor* tensor_dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
        Tensor* tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
        std::string binary_math_type = ctx->GetAttr<std::string>("binary_math_type");	
        
#define MATH_BINARY_BACKWARD(binary_math_type_str, func_name_prefix)                                       \
 if(binary_math_type == binary_math_type_str){                                                             \
	 func_name_prefix##Backward(ctx->device_ctx(), tensor_x, tensor_y, tensor_dz, tensor_dx, tensor_dy);   \
 }																				   
 
  OF_PP_FOR_EACH_TUPLE(MATH_BINARY_BACKWARD, MATH_BINARY_GPU_FLOAT_SEQ);
#undef MATH_BINARY_FORWARD
  }
};

REGISTER_USER_KERNEL("binary_grad")
    .SetCreateFn([](const KernelInitContext& ctx) { return new MathBinaryGradGpuFloatKernel(ctx); })
    .SetIsMatchedPred([](const KernelRegContext& ctx) {
      if (ctx.device() == DeviceType::kGPU && ctx.data_type() == DataType::kFloat) { return true; }
      return false;
    });

#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow

		