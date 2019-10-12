#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void UpsampleNearestForward(const int64_t nthreads, const T* in_dptr,
                                       const int64_t channel_num, const int64_t height,
                                       const int64_t width, const int64_t new_height,
                                       const int64_t new_width, const float scale_h,
                                       const float scale_w, const bool align_corners, T* out_dptr) {
  const int64_t new_area = new_height * new_width;
  const int64_t channel_area = channel_num * height * width;
  const int64_t channel_new_area = channel_num * new_height * new_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / new_width) % new_height;
    const int64_t w = index % new_width;
    const int64_t c = (index / new_area) % channel_num;
    const int64_t n = index / channel_new_area;
    const int64_t in_h = min((align_corners) ? static_cast<int64_t>(roundf(h * scale_h))
                                             : static_cast<int64_t>(floorf(h * scale_h)),
                             height - 1);
    const int64_t in_w = min((align_corners) ? static_cast<int64_t>(roundf(w * scale_w))
                                             : static_cast<int64_t>(floorf(w * scale_w)),
                             width - 1);
    out_dptr[index] = in_dptr[n * channel_area + (c * height + in_h) * width + in_w];
  }
}

template<typename T>
__global__ void UpsampleNearestBackward(const int64_t nthreads, const T* dy_dptr,
                                        const int64_t channel_num, const int64_t height,
                                        const int64_t width, const int64_t new_height,
                                        const int64_t new_width, const float scale_h,
                                        const float scale_w, const bool align_corners, T* dx_dptr) {
  const int64_t area = height * width;
  const int64_t new_area = new_height * new_width;
  const int64_t channel_area = channel_num * height * width;
  const int64_t channel_new_area = channel_num * new_height * new_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / new_width) % new_height;
    const int64_t w = index % new_width;
    const int64_t c = (index / new_area) % channel_num;
    const int64_t n = index / channel_new_area;
    const int64_t in_h = min((align_corners) ? static_cast<int64_t>(roundf(h * scale_h))
                                             : static_cast<int64_t>(floorf(h * scale_h)),
                             height - 1);
    const int64_t in_w = min((align_corners) ? static_cast<int64_t>(roundf(w * scale_w))
                                             : static_cast<int64_t>(floorf(w * scale_w)),
                             width - 1);
    gpu_atomic_add(dx_dptr + n * channel_area + (c * height + in_h) * width + in_w, dy_dptr[index]);
  }
}

}  // namespace

template<typename T>
class UpsampleNearestGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UpsampleNearestGPUKernel);
  UpsampleNearestGPUKernel() = default;
  ~UpsampleNearestGPUKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    const int32_t scale = this->op_conf().upsample_nearest_conf().scale();

    const int64_t elem_cnt = out_blob->shape().elem_cnt();
    UpsampleNearestForward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, in_blob->dptr<T>(), in_blob->shape().At(1), in_blob->shape().At(2),
        in_blob->shape().At(3), out_blob->shape().At(2), out_blob->shape().At(3), 1.f / scale,
        1.f / scale, false, out_blob->mut_dptr<T>());
  }
};

template<typename T>
class UpsampleNearestGradGPUKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UpsampleNearestGradGPUKernel);
  UpsampleNearestGradGPUKernel() = default;
  ~UpsampleNearestGradGPUKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* dx_blob = BnInOp2Blob("dx");
    if (dx_blob == nullptr) { return; }
    Memset<DeviceType::kGPU>(ctx.device_ctx, dx_blob->mut_dptr<T>(), 0,
                             dx_blob->ByteSizeOfBlobBody());
    const Blob* dy_blob = BnInOp2Blob("dy");
    const int32_t scale = this->op_conf().upsample_nearest_grad_conf().scale();

    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    UpsampleNearestBackward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                 ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, dy_blob->dptr<T>(), dx_blob->shape().At(1), dx_blob->shape().At(2),
        dx_blob->shape().At(3), dy_blob->shape().At(2), dy_blob->shape().At(3), 1.f / scale,
        1.f / scale, false, dx_blob->mut_dptr<T>());
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kUpsampleNearestConf, DeviceType::kGPU, float,
                                      UpsampleNearestGPUKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kUpsampleNearestGradConf, DeviceType::kGPU,
                                      float, UpsampleNearestGradGPUKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kUpsampleNearestConf, DeviceType::kGPU, double,
                                      UpsampleNearestGPUKernel<double>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kUpsampleNearestGradConf, DeviceType::kGPU,
                                      double, UpsampleNearestGradGPUKernel<double>)

}  // namespace oneflow
