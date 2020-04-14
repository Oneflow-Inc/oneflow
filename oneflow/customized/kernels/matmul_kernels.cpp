#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

template<typename T>
class MatmulGpuFloatingKernel final : public user_op::OpKernel {
 public:
  MatmulGpuFloatingKernel() = default;
  ~MatmulGpuFloatingKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    CBLAS_TRANSPOSE trans_a = ctx->GetAttr<bool>("transpose_a") ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = ctx->GetAttr<bool>("transpose_b") ? CblasTrans : CblasNoTrans;
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    int m, n, k;
    {
      m = out->shape().At(0);
      n = out->shape().At(1);
      k = (trans_a == CblasNoTrans) ? a->shape().At(1) : a->shape().At(0);
    }

    NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx->device_ctx(), trans_a, trans_b, m, n, k,
                                            GetOneVal<T>(), a->dptr<T>(), b->dptr<T>(),
                                            GetZeroVal<T>(), out->mut_dptr<T>());
  }
};

#define REGISTER_MATMUL_GPU_KERNEL(dtype)                                                        \
  REGISTER_USER_KERNEL("matmul").SetCreateFn<MatmulGpuFloatingKernel<dtype>>().SetIsMatchedPred( \
      [](const user_op::KernelRegContext& ctx) {                                                 \
        return ctx.device_type() == DeviceType::kGPU                                             \
               && ctx.TensorDesc4ArgNameAndIndex("a", 0)->data_type()                            \
                      == GetDataType<dtype>::value;                                              \
      })

REGISTER_MATMUL_GPU_KERNEL(float);
REGISTER_MATMUL_GPU_KERNEL(double);

REGISTER_FUNCTION_CONFIG_DEF().Bool(
    "enable_float_compute_for_half_gemm", false,
    "true means that the type of intermedia value is float when compute half gemm");

class MatmulGpuHalfKernel final : public user_op::OpKernel {
 public:
  MatmulGpuHalfKernel() = default;
  ~MatmulGpuHalfKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    CBLAS_TRANSPOSE trans_a = ctx->GetAttr<bool>("transpose_a") ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE trans_b = ctx->GetAttr<bool>("transpose_b") ? CblasTrans : CblasNoTrans;
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    int m, n, k;
    {
      m = out->shape().At(0);
      n = out->shape().At(1);
      k = (trans_a == CblasNoTrans) ? a->shape().At(1) : a->shape().At(0);
    }

    if (ctx->job_desc().Bool("enable_float_compute_for_half_gemm")) {
      NewKernelUtil<DeviceType::kGPU>::OFHGemmWithFloat(
          ctx->device_ctx(), trans_a, trans_b, m, n, k, GetOneVal<float>(), a->dptr<float16>(),
          b->dptr<float16>(), GetZeroVal<float>(), out->mut_dptr<float16>());
    } else {
      NewKernelUtil<DeviceType::kGPU>::OFHGemmWithFloat(
          ctx->device_ctx(), trans_a, trans_b, m, n, k, GetOneVal<float16>(), a->dptr<float16>(),
          b->dptr<float16>(), GetZeroVal<float16>(), out->mut_dptr<float16>());
    }
  }
};

REGISTER_USER_KERNEL("matmul").SetCreateFn<MatmulGpuHalfKernel>().SetIsMatchedPred(
    [](const user_op::KernelRegContext& ctx) {
      return ctx.device_type() == DeviceType::kGPU
             && ctx.TensorDesc4ArgNameAndIndex("a", 0)->data_type() == DataType::kFloat16;
    });

}  // namespace oneflow
