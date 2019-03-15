#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/transpose_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
void SoftmaxComputeDiff(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* dy, const T* out,
                        T* sum_vec, T* dx, void* temp_storage, const size_t temp_storage_bytes) {
  // it's safe to use dx as tmp
  // dot product | get dot product sum_vec[i] from out[i] * dy[i]
  T* tmp = dx;
  KernelUtil<device_type, T>::Mul(ctx, n * w, out, dy, tmp);
  NdarrayUtil<device_type, T>::ReduceSum(
      ctx, XpuVarNdarray<T>({n, 1}, sum_vec), XpuVarNdarray<const T>({n, w}, tmp),
      XpuVarNdarray<T>({static_cast<int64_t>(temp_storage_bytes / sizeof(T))},
                       reinterpret_cast<T*>(temp_storage)));
  // copy dy to dx
  KernelUtil<device_type, T>::Copy(ctx, n * w, dy, 1, dx, 1);
  // sub | dx[i][j] -= sum_vec[i]
  SoftmaxKernelUtil<device_type, T>::Sub(ctx, n, w, dx, sum_vec);
  // elementwise multiplication | dx[i][j] *= out[i][j]
  KernelUtil<device_type, T>::Mul(ctx, n * w, dx, out, dx);
}

}  // namespace

template<DeviceType device_type, typename T>
void SoftmaxComputeProb(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* x, T* tmp,
                        T* prob, void* temp_storage, const size_t temp_storage_bytes) {
  // copy x blob to prob blob
  KernelUtil<device_type, T>::Copy(ctx, n * w, x, 1, prob, 1);
  // max | calculate max of every sample vector prob[i], store in tmp[i]
  //       the prob[i] now is store the data of x[i]
  NdarrayUtil<device_type, T>::ReduceMax(
      ctx, XpuVarNdarray<T>({n, 1}, tmp), XpuVarNdarray<const T>({n, w}, prob),
      XpuVarNdarray<T>({static_cast<int64_t>(temp_storage_bytes / sizeof(T))},
                       reinterpret_cast<T*>(temp_storage)));
  // sub | every element of prob blob subract the max value of the same sample
  SoftmaxKernelUtil<device_type, T>::Sub(ctx, n, w, prob, tmp);
  // exp | exponentiation every element
  KernelUtil<device_type, T>::Exp(ctx, n * w, prob, prob);
  // sum | calculate sum of every sample vector prob[i], store in tmp[i]
  //       the prob[i] now is store the tmp data after exp
  NdarrayUtil<device_type, T>::ReduceSum(
      ctx, XpuVarNdarray<T>({n, 1}, tmp), XpuVarNdarray<const T>({n, w}, prob),
      XpuVarNdarray<T>({static_cast<int64_t>(temp_storage_bytes / sizeof(T))},
                       reinterpret_cast<T*>(temp_storage)));
  // div | every element of prob[i] divided by the data of tmp[i] (the sum
  // value)
  SoftmaxKernelUtil<device_type, T>::Div(ctx, n, w, prob, tmp);
}

template<DeviceType device_type, typename T>
void SoftmaxKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* x_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
  Blob* y_blob = BnInOp2Blob(this->op_attribute().output_bns(0));
  Blob* tmp_blob = BnInOp2Blob("fw_softmax_num");
  Blob* buf_blob = BnInOp2Blob("fw_buf");
  auto conf = this->kernel_conf().softmax_conf();
  const int64_t n = conf.transpose_rows();
  const int64_t w = conf.transpose_cols();
  T* tmp = tmp_blob->mut_dptr<T>();
  if (conf.need_transpose()) {
    Blob* transpose_x_blob = BnInOp2Blob("transpose_x");
    Blob* transpose_y_blob = BnInOp2Blob("transpose_y");
    Transpose<device_type, T>(ctx.device_ctx, x_blob, transpose_x_blob, conf.perm());
    SoftmaxComputeProb<device_type, T>(ctx.device_ctx, n, w, transpose_x_blob->dptr<T>(), tmp,
                                       transpose_y_blob->mut_dptr<T>(), buf_blob->mut_dptr(),
                                       buf_blob->ByteSizeOfDataContentField());
    Transpose<device_type, T>(ctx.device_ctx, transpose_y_blob, y_blob, conf.perm());
  } else {
    SoftmaxComputeProb<device_type, T>(ctx.device_ctx, n, w, x_blob->dptr<T>(), tmp,
                                       y_blob->mut_dptr<T>(), buf_blob->mut_dptr(),
                                       buf_blob->ByteSizeOfDataContentField());
  }
}

template<DeviceType device_type, typename T>
void SoftmaxKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* y_blob = BnInOp2Blob(this->op_attribute().output_bns(0));
  const Blob* dy_blob = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
  Blob* dx_blob = BnInOp2Blob(this->op_attribute().input_diff_bns(0));
  Blob* tmp_blob = BnInOp2Blob("bw_softmax_num");
  Blob* buf_blob = BnInOp2Blob("bw_buf");
  auto conf = this->kernel_conf().softmax_conf();
  const int64_t n = conf.transpose_rows();
  const int64_t w = conf.transpose_cols();
  T* tmp = tmp_blob->mut_dptr<T>();
  if (conf.need_transpose()) {
    Blob* transpose_dx_blob = BnInOp2Blob("transpose_x");
    Blob* transpose_y_blob = BnInOp2Blob("transpose_y");
    Blob* transpose_dy_blob = BnInOp2Blob("transpose_dy");
    Transpose<device_type, T>(ctx.device_ctx, dy_blob, transpose_dy_blob, conf.perm());
    SoftmaxComputeDiff<device_type, T>(ctx.device_ctx, n, w, transpose_dy_blob->dptr<T>(),
                                       transpose_y_blob->dptr<T>(), tmp,
                                       transpose_dx_blob->mut_dptr<T>(), buf_blob->mut_dptr(),
                                       buf_blob->ByteSizeOfDataContentField());
    Transpose<device_type, T>(ctx.device_ctx, transpose_dx_blob, dx_blob, conf.perm());
  } else {
    SoftmaxComputeDiff<device_type, T>(ctx.device_ctx, n, w, dy_blob->dptr<T>(), y_blob->dptr<T>(),
                                       tmp, dx_blob->mut_dptr<T>(), buf_blob->mut_dptr(),
                                       buf_blob->ByteSizeOfDataContentField());
  }
}

template<typename T>
struct SoftmaxKernelUtil<DeviceType::kCPU, T> {
  static void Sub(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix, const T* vector) {
    for (int64_t i = 0; i < w; ++i) {
      KernelUtil<DeviceType::kCPU, T>::Axpy(ctx, n, static_cast<T>(-1.0), vector, 1, matrix + i, w);
    }
  }

  static void Div(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix, const T* vector) {
    for (int64_t i = 0; i < n; ++i) {
      KernelUtil<DeviceType::kCPU, T>::Div(ctx, w, matrix + i * w, vector + i);
    }
  }
};
#define INSTANTIATE_SOFTMAX_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SoftmaxKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SOFTMAX_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSoftmaxConf, SoftmaxKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
