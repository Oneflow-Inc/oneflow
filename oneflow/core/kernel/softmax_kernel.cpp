#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void SoftmaxKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(op()->SoleIbn());
  Blob* out_blob = BnInOp2Blob(op()->SoleObn());
  Blob* tmp_blob = BnInOp2Blob(op()->SoleDtbn());
  const int64_t n = out_blob->shape().At(0);
  const int64_t w = out_blob->shape().At(1);
  const T* in = in_blob->dptr<T>();
  T* tmp = tmp_blob->mut_dptr<T>();
  T* out = out_blob->mut_dptr<T>();
  CopyDataIdFromSoleIbToAllObIfNeed<device_type>(ctx, BnInOp2Blob);
  SoftmaxComputeProb<device_type, T>(ctx.device_ctx, n, w, in, tmp, out);
}

template<DeviceType device_type, typename T>
void SoftmaxKernel<device_type, T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_blob = BnInOp2Blob(op()->SoleObn());
  const Blob* out_diff_blob = BnInOp2Blob(op()->SoleOdbn());
  Blob* in_diff_blob = BnInOp2Blob(op()->SoleIdbn());
  Blob* tmp_blob = BnInOp2Blob(op()->SoleDtbn());
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

Kernel* CreateSoftmaxKernel(const OpContext& op_ctx) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define SOFTMAX_KERNEL_ENTRY(device_type, data_type_pair)                     \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)), []() {         \
     return new SoftmaxKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>; \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SOFTMAX_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ)};

  return creators.at(
      GetHashKey(op_ctx.device_type(), op_ctx.bn_in_op2data_type().at("in")))();
}

COMMAND(AddKernelCreator(OperatorConf::kSoftmaxConf, CreateSoftmaxKernel));

}  // namespace oneflow
