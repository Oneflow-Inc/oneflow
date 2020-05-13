#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void gpu_assign_add(const int64_t n, T* out, const T* in_1) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (in_1[i]) { out[i] += in_1[i]; }
  }
}

template<typename T>
__global__ void gpu_assign_add(const int64_t n, T* out, const T* in_1, const T* in_2) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] += in_1[i] + in_2[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i]; }
}
template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i]; }
}

}  // namespace

template<typename T>
class GpuAddNKernel : public user_op::OpKernel {
 public:
  GpuAddNKernel() = default;
  ~GpuAddNKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    size_t in_num = ctx->inputs().size();
    CHECK_GE(in_num, 2);
    CHECK_LE(in_num, 4)
        << "GpuAddNKernel of add_n op doesn't support number of operands which is bigger than 4.";

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t n = out->shape().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();

    std::vector<const T*> in_dptrs(in_num);
    for (int32_t i = 0; i < in_num; ++i) {
      in_dptrs.at(i) = ctx->Tensor4ArgNameAndIndex("in", i)->dptr<T>();
    }

    if (in_num == 2) {
      if (out_dptr == in_dptrs.at(0)) {
        gpu_assign_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                            ctx->device_ctx()->cuda_stream()>>>(n, out_dptr, in_dptrs.at(1));
      } else {
        gpu_add<T>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
               ctx->device_ctx()->cuda_stream()>>>(n, out_dptr, in_dptrs.at(0), in_dptrs.at(1));
      }
    } else if (in_num == 3) {
      if (out_dptr == in_dptrs.at(0)) {
        gpu_assign_add<T>
            <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
               ctx->device_ctx()->cuda_stream()>>>(n, out_dptr, in_dptrs.at(1), in_dptrs.at(2));
      } else {
        gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                     ctx->device_ctx()->cuda_stream()>>>(n, out_dptr, in_dptrs.at(0),
                                                         in_dptrs.at(1), in_dptrs.at(2));
      }
    } else if (in_num == 4) {
      gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                   ctx->device_ctx()->cuda_stream()>>>(n, out_dptr, in_dptrs.at(0), in_dptrs.at(1),
                                                       in_dptrs.at(2), in_dptrs.at(3));
    } else {
      LOG(FATAL) << "Not supported input size for GpuAddNKernel: " << in_num;
    }
  }
};

#define REGISTER_GPU_ADDN_KERNEL(cpp_type, dtype)                                        \
  REGISTER_USER_KERNEL("add_n").SetCreateFn<GpuAddNKernel<cpp_type>>().SetIsMatchedPred( \
      [](const user_op::KernelRegContext& ctx) {                                         \
        return ctx.device_type() == DeviceType::kGPU                                     \
               && ctx.TensorDesc4ArgNameAndIndex("in", 0)->data_type() == dtype;         \
      });

OF_PP_FOR_EACH_TUPLE(REGISTER_GPU_ADDN_KERNEL, ARITHMETIC_DATA_TYPE_SEQ);

namespace {

__global__ void half_gpu_add(const int64_t n, half* out, const half* in_0, const half* in_1) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = __hadd(in_0[i], in_1[i]); }
}

__global__ void half_gpu_add(const int64_t n, half* out, const half* in_0, const half* in_1,
                             const half* in_2) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = __hadd(in_0[i], __hadd(in_1[i], in_2[i])); }
}

__global__ void half_gpu_add(const int64_t n, half* out, const half* in_0, const half* in_1,
                             const half* in_2, const half* in_3) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = __hadd(in_0[i], __hadd(in_1[i], __hadd(in_2[i], in_3[i]))); }
}

}  // namespace

class GpuAddNHalfKernel : public user_op::OpKernel {
 public:
  GpuAddNHalfKernel() = default;
  ~GpuAddNHalfKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    size_t in_num = ctx->inputs().size();
    CHECK_GE(in_num, 2);
    CHECK_LE(in_num, 4) << "GpuAddNHalfKernel of add_n op doesn't support number of operands which "
                           "is bigger than 4.";

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t n = out->shape().elem_cnt();

    half* half_out_dptr = reinterpret_cast<half*>(out->mut_dptr<float16>());
    std::vector<const half*> half_in_dptrs(in_num);
    for (int32_t i = 0; i < in_num; ++i) {
      half_in_dptrs.at(i) =
          reinterpret_cast<const half*>(ctx->Tensor4ArgNameAndIndex("in", i)->dptr<float16>());
    }
    switch (half_in_dptrs.size()) {
      case 2:
        half_gpu_add<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                       ctx->device_ctx()->cuda_stream()>>>(n, half_out_dptr, half_in_dptrs.at(0),
                                                           half_in_dptrs.at(1));
        break;
      case 3:
        half_gpu_add<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                       ctx->device_ctx()->cuda_stream()>>>(
            n, half_out_dptr, half_in_dptrs.at(0), half_in_dptrs.at(1), half_in_dptrs.at(2));
        break;
      case 4:
        half_gpu_add<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                       ctx->device_ctx()->cuda_stream()>>>(n, half_out_dptr, half_in_dptrs.at(0),
                                                           half_in_dptrs.at(1), half_in_dptrs.at(2),
                                                           half_in_dptrs.at(3));
        break;
      default: LOG(FATAL) << "Not supported input size for GpuAddNHalfKernel: " << in_num; break;
    }
  }
};

REGISTER_USER_KERNEL("add_n").SetCreateFn<GpuAddNHalfKernel>().SetIsMatchedPred(
    [](const user_op::KernelRegContext& ctx) {
      return ctx.device_type() == DeviceType::kGPU
             && ctx.TensorDesc4ArgNameAndIndex("in", 0)->data_type() == DataType::kFloat16;
    });

}  // namespace oneflow
