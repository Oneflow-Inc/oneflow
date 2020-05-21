#include "oneflow/core/kernel/softmax_kernel.h"
#include "oneflow/core/kernel/softmax_grad_kernel.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/transpose_kernel.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
void SoftmaxComputeDiff(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* dy, const T* out,
                        T* sum_vec, T* dx, void* temp_storage, const size_t temp_storage_bytes) {
  auto Val = NdarrayUtil<device_type, T>::GetValNdarrayBuilder();
  auto Var = NdarrayUtil<device_type, T>::GetVarNdarrayBuilder();
  // it's safe to use dx as tmp
  // dot product | get dot product sum_vec[i] from out[i] * dy[i]
  T* tmp = dx;
  NdarrayUtil<device_type, T>::Mul(ctx, Var({n * w}, tmp), Val({n * w}, out), Val({n * w}, dy));
  NdarrayUtil<device_type, T>::ReduceSum(ctx, Var({n, 1}, sum_vec), Val({n, w}, tmp),
                                         Var({static_cast<int64_t>(temp_storage_bytes / sizeof(T))},
                                             reinterpret_cast<T*>(temp_storage)));
  // sub | dx[i][j] = dy[i][j] - sum_vec[i]
  NdarrayUtil<device_type, T>::BroadcastSub(ctx, Var({n, w}, dx), Val({n, w}, dy),
                                            Val({n, 1}, sum_vec));
  // elementwise multiplication | dx[i][j] *= out[i][j]
  NdarrayUtil<device_type, T>::InplaceMul(ctx, Var({n * w}, dx), Val({n * w}, out));
}

}  // namespace

template<DeviceType device_type, typename T>
void SoftmaxGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* y_blob = BnInOp2Blob("y");
  const Blob* dy_blob = BnInOp2Blob("dy");
  Blob* dx_blob = BnInOp2Blob("dx");
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
    SoftmaxComputeDiff<device_type, T>(
        ctx.device_ctx, n, w, transpose_dy_blob->dptr<T>(), transpose_y_blob->dptr<T>(), tmp,
        transpose_dx_blob->mut_dptr<T>(), buf_blob->mut_dptr(), buf_blob->ByteSizeOfBlobBody());
    Transpose<device_type, T>(ctx.device_ctx, transpose_dx_blob, dx_blob, conf.perm());
  } else {
    SoftmaxComputeDiff<device_type, T>(ctx.device_ctx, n, w, dy_blob->dptr<T>(), y_blob->dptr<T>(),
                                       tmp, dx_blob->mut_dptr<T>(), buf_blob->mut_dptr(),
                                       buf_blob->ByteSizeOfBlobBody());
  }
}

ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kSoftmaxGradConf, SoftmaxGradKernel,
                                         FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
