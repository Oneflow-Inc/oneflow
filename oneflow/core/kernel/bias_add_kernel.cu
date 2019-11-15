#include "oneflow/core/kernel/bias_add_kernel.h"

namespace oneflow {

namespace {

template<typename T, typename Index>
__global__ void BiasAddGpu(const Index elem_cnt, const Index bias_size, const Index inner_size,
                           const T* x, const T* bias, T* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) { y[i] = x[i] + bias[(i % block_size) / inner_size]; }
}

template<typename Index>
__global__ void BiasAddForwardGpuHalf(const Index elem_cnt, const Index bias_size,
                                      const Index inner_size, const half* x, const half* bias,
                                      half* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) {
    y[i] = __hadd(x[i], bias[(i % block_size) / inner_size]);
  }
}

template<typename T, typename Index>
__global__ void InplaceBiasAddGpu(const Index elem_cnt, const Index bias_size,
                                  const Index inner_size, const T* bias, T* y) {
  const Index block_size = bias_size * inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, elem_cnt) { y[i] += bias[(i % block_size) / inner_size]; }
}

template<typename T, typename Index>
struct BiasAddGpuHelper final {
  static void BiasAdd(DeviceCtx* ctx, const Index elem_cnt, const Index bias_size,
                      const Index inner_size, const T* x, const T* bias, T* y) {
    if (x == y) {
      InplaceBiasAddGpu<T, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, bias_size, inner_size, bias, y);
    } else {
      BiasAddGpu<T, Index>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, bias_size, inner_size, x, bias, y);
    }
  }
};

}  // namespace

template<typename T>
struct BiasAddUtil<DeviceType::kGPU, T> {
  static void BiasAdd(DeviceCtx* ctx, int64_t outer_size, int64_t bias_size, int64_t inner_size,
                      const T* x, const T* bias, T* y) {
    const int64_t elem_cnt = outer_size * bias_size * inner_size;
    if (IsKernelSafeInt32(elem_cnt)) {
      BiasAddGpuHelper<T, int32_t>::BiasAdd(ctx, elem_cnt, bias_size, inner_size, x, bias, y);
    } else {
      BiasAddGpuHelper<T, int64_t>::BiasAdd(ctx, elem_cnt, bias_size, inner_size, x, bias, y);
    }
  }
};

class BiasAddGpuHalfKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BiasAddGpuHalfKernel);
  BiasAddGpuHalfKernel() = default;
  ~BiasAddGpuHalfKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* a_blob = BnInOp2Blob("a");
    const Blob* b_blob = BnInOp2Blob("b");
    Blob* out_blob = BnInOp2Blob("out");
    const BiasAddOpConf& conf = this->op_conf().bias_add_conf();
    const int32_t bias_add_axis = conf.axis();
    const int64_t outer_size = a_blob->shape().Count(0, bias_add_axis);
    const int64_t bias_size = a_blob->shape().At(bias_add_axis);
    const int64_t inner_size = a_blob->shape().Count(bias_add_axis + 1);
    const int64_t elem_cnt = outer_size * bias_size * inner_size;
    if (IsKernelSafeInt32(elem_cnt)) {
      BiasAddForwardGpuHalf<int32_t><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                       ctx.device_ctx->cuda_stream()>>>(
          elem_cnt, bias_size, inner_size, reinterpret_cast<const half*>(a_blob->dptr<float16>()),
          reinterpret_cast<const half*>(b_blob->dptr<float16>()),
          reinterpret_cast<half*>(out_blob->mut_dptr<float16>()));

    } else {
      BiasAddForwardGpuHalf<int64_t><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                       ctx.device_ctx->cuda_stream()>>>(
          elem_cnt, bias_size, inner_size, reinterpret_cast<const half*>(a_blob->dptr<float16>()),
          reinterpret_cast<const half*>(b_blob->dptr<float16>()),
          reinterpret_cast<half*>(out_blob->mut_dptr<float16>()));
    }
  }

  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().bias_add_conf(); }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBiasAddConf, DeviceType::kGPU, float16,
                                      BiasAddGpuHalfKernel)

#define INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL(type_cpp, type_proto) \
  template struct BiasAddUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL, ARITHMETIC_DATA_TYPE_SEQ);
#undef INITIATE_BIAS_ADD_KERNEL_UTIL_GPU_IMPL

}  // namespace oneflow
