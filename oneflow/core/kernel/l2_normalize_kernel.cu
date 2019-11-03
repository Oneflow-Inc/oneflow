#include "oneflow/core/kernel/kernel.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

template<typename T>
__global__ void L2NormalizeForward(const int32_t n, const int32_t c, const int32_t d,
                                   const T epsilon, const T* in, T* square_x_sum, T* out) {
  using BlockReduce = cub::BlockReduce<T, kCudaThreadsNumPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int32_t i = blockIdx.x; i < n; i += gridDim.x) {
    T sum = GetZeroVal<T>();
    const int32_t offset = (i / d) * d * c + (i % d);
    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const T x = in[offset + j * d];
      sum += x * x;
    }
    const T reduce_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) { square_x_sum[i] = reduce_sum; }
    __syncthreads();

    const T inv_norm = rsqrtf(fmaxf(square_x_sum[i], epsilon));
    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const int32_t index = offset + j * d;
      out[index] = inv_norm * in[index];
    }
  }
}

template<typename T>
__global__ void L2NormalizeBackward(const int32_t n, const int32_t c, const int32_t d,
                                    const float epsilon, const T* out, const T* out_diff,
                                    const T* square_x_sum, T* in_diff) {
  for (int32_t i = blockIdx.x; i < n; i += gridDim.x) {
    const T inv_norm = rsqrt(fmaxf(square_x_sum[i], epsilon));
    const int32_t offset = (i / d) * d * c + (i % d);
    if (square_x_sum[i] >= epsilon) {
      using BlockReduce = cub::BlockReduce<T, kCudaThreadsNumPerBlock>;
      __shared__ typename BlockReduce::TempStorage temp_storage_prod_sum;

      T y_dy_prod_sum = GetZeroVal<T>();
      for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
        const int32_t index = offset + j * d;
        y_dy_prod_sum += out[index] * out_diff[index];
      }

      const T reduce_y_dy_prod_sum = BlockReduce(temp_storage_prod_sum).Sum(y_dy_prod_sum);
      __shared__ T y_dy_inner_prod;
      if (threadIdx.x == 0) { y_dy_inner_prod = reduce_y_dy_prod_sum; }
      __syncthreads();

      for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
        const int32_t index = offset + j * d;
        in_diff[index] = inv_norm * (out_diff[index] - y_dy_inner_prod * out[index]);
      }
    } else {
      for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
        const int32_t index = offset + j * d;
        in_diff[index] = inv_norm * out_diff[index];
      }
    }
  }
}

}  // namespace

template<typename T>
class L2NormalizeGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormalizeGpuKernel);
  L2NormalizeGpuKernel() = default;
  ~L2NormalizeGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    const int32_t axis = this->op_conf().l2_normalize_conf().axis();
    const float epsilon = this->op_conf().l2_normalize_conf().epsilon();
    int32_t c = in_blob->shape().At(axis);
    int32_t n = in_blob->shape().elem_cnt() / c;
    int32_t d = in_blob->shape().Count(axis + 1);
    L2NormalizeForward<<<std::min(n, kCudaMaxBlocksNum), kCudaThreadsNumPerBlock, 0,
                         ctx.device_ctx->cuda_stream()>>>(
        n, c, d, static_cast<T>(epsilon), in_blob->dptr<T>(),
        BnInOp2Blob("square_x_sum")->mut_dptr<T>(), BnInOp2Blob("out")->mut_dptr<T>());
  }
};

template<typename T>
class L2NormalizeGradGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormalizeGradGpuKernel);
  L2NormalizeGradGpuKernel() = default;
  ~L2NormalizeGradGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* dy_blob = BnInOp2Blob("dy");
    const int32_t axis = this->op_conf().l2_normalize_grad_conf().axis();
    const float epsilon = this->op_conf().l2_normalize_grad_conf().epsilon();
    int32_t c = dy_blob->shape().At(axis);
    int32_t n = dy_blob->shape().elem_cnt() / c;
    int32_t d = dy_blob->shape().Count(axis + 1);
    L2NormalizeBackward<<<std::min(n, kCudaMaxBlocksNum), kCudaThreadsNumPerBlock, 0,
                          ctx.device_ctx->cuda_stream()>>>(
        n, c, d, static_cast<T>(epsilon), BnInOp2Blob("y")->dptr<T>(), dy_blob->dptr<T>(),
        BnInOp2Blob("square_x_sum")->dptr<T>(), BnInOp2Blob("dx")->mut_dptr<T>());
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kL2NormalizeConf, DeviceType::kGPU, float,
                                      L2NormalizeGpuKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kL2NormalizeConf, DeviceType::kGPU, double,
                                      L2NormalizeGpuKernel<double>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kL2NormalizeGradConf, DeviceType::kGPU, float,
                                      L2NormalizeGradGpuKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kL2NormalizeGradConf, DeviceType::kGPU, double,
                                      L2NormalizeGradGpuKernel<double>)

}  // namespace oneflow
