#include "oneflow/core/kernel/boxing_align_broadcast_reshape_kernel.h"

#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BoxingAlignBroadcastReshapeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& kernel_ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const BoxingAlignBroadcastReshapeOpConf& conf =
      this->op_conf().boxing_align_broadcast_reshape_conf();
  CHECK_GE(conf.target_parallel_num(), 1);
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  CHECK_EQ(out->shape().elem_cnt() % conf.target_parallel_num(), 0);
  const int64_t aligned_elem_cnt = out->shape().elem_cnt() / conf.target_parallel_num();
  CHECK_LE(in->shape().elem_cnt(), aligned_elem_cnt);
  const int64_t aligned_byte_size = aligned_elem_cnt * GetSizeOfDataType(out->data_type());
  FOR_RANGE(int64_t, i, 0, conf.target_parallel_num()) {
    Memcpy<device_type>(kernel_ctx.device_ctx, out->mut_dptr<char>() + i * aligned_byte_size,
                        in->dptr(), in->ByteSizeOfDataContentField());
  }
}

template<DeviceType device_type, typename T>
void BoxingAlignBroadcastReshapeKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& kernel_ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const BoxingAlignBroadcastReshapeOpConf& conf =
      this->op_conf().boxing_align_broadcast_reshape_conf();
  CHECK_GE(conf.target_parallel_num(), 1);
  const Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff = BnInOp2Blob(GenDiffBn("in"));
  Blob* bw_buf = BnInOp2Blob("bw_buf");
  CHECK_LE(in_diff->shape().elem_cnt(), bw_buf->shape().elem_cnt());
  Blob* reduce_buf = BnInOp2Blob("reduce_buf");
  const int64_t aligned_elem_cnt = bw_buf->shape().elem_cnt();
  CHECK_EQ(out_diff->shape().elem_cnt(), aligned_elem_cnt * conf.target_parallel_num());
  NdarrayUtil<device_type, T>::ReduceSum(
      kernel_ctx.device_ctx, XpuVarNdarray<T>({1, aligned_elem_cnt}, bw_buf->mut_dptr<T>()),
      XpuVarNdarray<const T>({conf.target_parallel_num(), aligned_elem_cnt}, out_diff->dptr<T>()),
      XpuVarNdarray<T>({conf.target_parallel_num(), aligned_elem_cnt}, reduce_buf->mut_dptr<T>()));
  in_diff->CopyDataContentFrom(kernel_ctx.device_ctx, bw_buf);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBoxingAlignBroadcastReshapeConf,
                           BoxingAlignBroadcastReshapeKernel, FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
