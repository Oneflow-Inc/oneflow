#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/transpose_kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
void SoftmaxComputeDiff(DeviceCtx* ctx, const int64_t n, const int64_t w,
                        const T* out_diff, const T* out, T* tmp, T* in_diff) {
  // copy out_diff to in_diff
  KernelUtil<device_type, T>::Copy(ctx, n * w, out_diff, 1, in_diff, 1);
  // dot product | get dot product tmp[i] from out[i] * out_diff[i]
  SoftmaxKernelUtil<device_type, T>::BackwardDot(ctx, n, w, out, out_diff, tmp);
  // sub | in_diff[i][j] -= tmp[i]
  SoftmaxKernelUtil<device_type, T>::Sub(ctx, n, w, in_diff, tmp);
  // elementwise multiplication | in_diff[i][j] *= out[i][j]
  KernelUtil<device_type, T>::Mul(ctx, n * w, in_diff, out, in_diff);
}

}  // namespace

template<DeviceType device_type, typename T>
void SoftmaxKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(this->kernel_conf().input_bns(0));
  Blob* out_blob = BnInOp2Blob(this->kernel_conf().output_bns(0));
  Blob* tmp_blob = BnInOp2Blob("softmax_num");
  auto conf = this->kernel_conf().softmax_conf();
  const int64_t n = conf.transpose_rows();
  const int64_t w = conf.transpose_cols();
  T* tmp = tmp_blob->mut_dptr<T>();
  if (conf.need_transpose()) {
    Blob* transpose_in_blob = BnInOp2Blob("transpose_in");
    Blob* transpose_out_blob = BnInOp2Blob("transpose_out");
    Transpose<device_type, T>(ctx.device_ctx, in_blob, transpose_in_blob,
                              conf.perm());
    SoftmaxComputeProb<device_type, T>(ctx.device_ctx, n, w,
                                       transpose_in_blob->dptr<T>(), tmp,
                                       transpose_out_blob->mut_dptr<T>());
    Transpose<device_type, T>(ctx.device_ctx, transpose_out_blob, out_blob,
                              conf.perm());
  } else {
    SoftmaxComputeProb<device_type, T>(ctx.device_ctx, n, w, in_blob->dptr<T>(),
                                       tmp, out_blob->mut_dptr<T>());
  }
}

template<DeviceType device_type, typename T>
void SoftmaxKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_blob = BnInOp2Blob(this->kernel_conf().output_bns(0));
  const Blob* out_diff_blob =
      BnInOp2Blob(this->kernel_conf().output_diff_bns(0));
  Blob* in_diff_blob = BnInOp2Blob(this->kernel_conf().input_diff_bns(0));
  Blob* tmp_blob = BnInOp2Blob("softmax_num");
  auto conf = this->kernel_conf().softmax_conf();
  const int64_t n = conf.transpose_rows();
  const int64_t w = conf.transpose_cols();
  T* tmp = tmp_blob->mut_dptr<T>();
  if (conf.need_transpose()) {
    Blob* transpose_in_diff_blob = BnInOp2Blob("transpose_in");
    Blob* transpose_out_blob = BnInOp2Blob("transpose_out");
    Blob* transpose_out_diff_blob = BnInOp2Blob("transpose_out_diff");
    Transpose<device_type, T>(ctx.device_ctx, out_diff_blob,
                              transpose_out_diff_blob, conf.perm());
    SoftmaxComputeDiff<device_type, T>(ctx.device_ctx, n, w,
                                       transpose_out_diff_blob->dptr<T>(),
                                       transpose_out_blob->dptr<T>(), tmp,
                                       transpose_in_diff_blob->mut_dptr<T>());
    Transpose<device_type, T>(ctx.device_ctx, transpose_in_diff_blob,
                              in_diff_blob, conf.perm());
  } else {
    SoftmaxComputeDiff<device_type, T>(
        ctx.device_ctx, n, w, out_diff_blob->dptr<T>(), out_blob->dptr<T>(),
        tmp, in_diff_blob->mut_dptr<T>());
  }
}

template<typename T>
struct SoftmaxKernelUtil<DeviceType::kCPU, T> {
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
      KernelUtil<DeviceType::kCPU, T>::Axpy(ctx, n, static_cast<T>(-1.0),
                                            vector, 1, matrix + i, w);
    }
  }

  static void BackwardDot(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const T* out, const T* out_diff, T* tmp) {
    for (int64_t i = 0; i < n; ++i) {
      KernelUtil<DeviceType::kCPU, T>::Dot(ctx, w, out + i * w, 1,
                                           out_diff + i * w, 1, tmp + i);
    }
  }
};
#define INSTANTIATE_SOFTMAX_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SoftmaxKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SOFTMAX_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSoftmaxConf, SoftmaxKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
