#include "oneflow/core/kernel/local_gpu_peer_partial_sum_to_split_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

#ifdef WITH_CUDA

template<typename T>
void LocalGpuPeerPartialSumToSplitKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out = BnInOp2Blob("out");
  Blob* buf = BnInOp2Blob("buf");
  const int64_t out_split_axis =
      op_conf().local_gpu_peer_partial_sum_to_split_conf().out_split_axis();
  const LocalGpuPeerPartialSumToSplitKernelConf& conf =
      kernel_conf().local_gpu_peer_partial_sum_to_split_conf();
  FOR_RANGE(int32_t, i, 0, op_attribute().input_bns_size()) {
    const std::string& ibn = op_attribute().input_bns(i);
    CHECK_EQ(ibn, GenRepeatedBn("in", i));
    const Blob* in_i = BnInOp2Blob(ibn);
    const RangeProto& range = conf.range();
    CHECK_LT(range.begin(), range.end());
    CHECK_LE(range.end(), in_i->shape().At(out_split_axis));
    CHECK_EQ(out->shape().At(out_split_axis), range.end() - range.begin());
    Blob* copy_to = i == 0 ? out : buf;
    if (out_split_axis == 0) {
      CudaCheck(cudaMemcpyAsync(copy_to->mut_dptr<T>(), in_i->dptr<T>(range.begin()),
                                copy_to->ByteSizeOfDataContentField(), cudaMemcpyDefault,
                                ctx.device_ctx->cuda_stream()));
    } else {
      const int64_t rows = out->shape().Count(0, out_split_axis);
      const ino64_t inner_size = out->shape().Count(out_split_axis + 1) * sizeof(T);
      const int64_t src_pitch = in_i->shape().At(out_split_axis) * inner_size;
      const void* src_dptr = in_i->dptr<char>() + range.begin() * inner_size;
      const int64_t dst_pitch = out->shape().At(out_split_axis) * inner_size;
      void* dst_dptr = out->mut_dptr();
      CudaCheck(cudaMemcpy2DAsync(dst_dptr, dst_pitch, src_dptr, src_pitch, dst_pitch, rows,
                                  cudaMemcpyDefault, ctx.device_ctx->cuda_stream()));
    }
    if (i != 0) {
      Addition<DeviceType::kGPU, T>(ctx.device_ctx, out->shape().elem_cnt(), out->mut_dptr<T>(),
                                    out->dptr<T>(), copy_to->dptr<T>());
    }
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalGpuPeerPartialSumToSplitConf,
                               LocalGpuPeerPartialSumToSplitKernel, ARITHMETIC_DATA_TYPE_SEQ)

#endif

}  // namespace oneflow
