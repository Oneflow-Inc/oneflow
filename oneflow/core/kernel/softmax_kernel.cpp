#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SoftmaxKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(this->kernel_conf().input_bns(0));
  Blob* out_blob = BnInOp2Blob(this->kernel_conf().output_bns(0));
  Blob* tmp_blob = BnInOp2Blob(this->kernel_conf().data_tmp_bns(0));
  const int64_t n = out_blob->shape().At(0);
  const int64_t w = out_blob->shape().At(1);
  const T* in = in_blob->dptr<T>();
  T* tmp = tmp_blob->mut_dptr<T>();
  T* out = out_blob->mut_dptr<T>();
  SoftmaxComputeProb<device_type, T>(ctx.device_ctx, n, w, in, tmp, out);
}

template<DeviceType device_type, typename T>
void SoftmaxKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_blob = BnInOp2Blob(this->kernel_conf().output_bns(0));
  const Blob* out_diff_blob =
      BnInOp2Blob(this->kernel_conf().output_diff_bns(0));
  Blob* in_diff_blob = BnInOp2Blob(this->kernel_conf().input_diff_bns(0));
  Blob* tmp_blob = BnInOp2Blob(this->kernel_conf().data_tmp_bns(0));
  const int64_t n = out_blob->shape().At(0);
  const int64_t w = out_blob->shape().At(1);
  T* in_diff = in_diff_blob->mut_dptr<T>();
  T* tmp = tmp_blob->mut_dptr<T>();
  const T* out = out_blob->dptr<T>();
  const T* out_diff = out_diff_blob->dptr<T>();
  // copy out_diff to in_diff
  KernelUtil<device_type, T>::BlasCopy(ctx.device_ctx, n * w, out_diff, 1,
                                       in_diff, 1);
  // dot product | get dot product tmp[i] from out[i] * out_diff[i]
  SoftmaxKernelUtil<device_type, T>::BackwardDot(ctx.device_ctx, n, w, out,
                                                 out_diff, tmp);
  // sub | in_diff[i][j] -= tmp[i]
  SoftmaxKernelUtil<device_type, T>::Sub(ctx.device_ctx, n, w, in_diff, tmp);
  // elementwise multiplication | in_diff[i][j] *= out[i][j]
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, n * w, in_diff, out, in_diff);
}

template<typename T>
class SoftmaxKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxKernelUtil);
  SoftmaxKernelUtil() = delete;

  static void ForwardMax(DeviceCtx* ctx, const int64_t n, const int64_t w,
                         const T* out, T* tmp) {
    for (int64_t i = 0; i < n; ++i) {
      KernelUtil<DeviceType::kCPU, T>::Max(ctx, w, out + i * w, tmp + i);
    }
  }

  static void ForwardSum(DeviceCtx* ctx, const int64_t n, const int64_t w,
                         const T* out, T* tmp) {
    for (int64_t i = 0; i < n; ++i) {
      KernelUtil<DeviceType::kCPU, T>::Sum(ctx, w, out + i * w, tmp + i);
    }
  }

  static void Sub(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix,
                  const T* vector) {
    for (int64_t i = 0; i < w; ++i) {
      KernelUtil<DeviceType::kCPU, T>::BlasAxpy(ctx, n, static_cast<T>(-1.0),
                                                vector, 1, matrix + i, w);
    }
  }

  static void BackwardDot(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const T* out, const T* out_diff, T* tmp) {
    for (int64_t i = 0; i < n; ++i) {
      KernelUtil<DeviceType::kCPU, T>::BlasDot(ctx, w, out + i * w, 1,
                                               out_diff + i * w, 1, tmp + i);
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSoftmaxConf, SoftmaxKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
