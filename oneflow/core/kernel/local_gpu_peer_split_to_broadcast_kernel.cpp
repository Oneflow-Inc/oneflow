#include "oneflow/core/kernel/local_gpu_peer_split_to_broadcast_kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

template<typename T>
void LocalGpuPeerSplitToBroadcastKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out = BnInOp2Blob("out");
  const int64_t in_split_axis = op_conf().local_gpu_peer_split_to_broadcast_conf().in_split_axis();
  const LocalGpuPeerSplitToBroadcastKernelConf& conf =
      kernel_conf().local_gpu_peer_split_to_broadcast_conf();
  FOR_RANGE(int32_t, i, 0, op_attribute().input_bns_size()) {
    const std::string& ibn = op_attribute().input_bns(i);
    CHECK_EQ(ibn, GenRepeatedBn("in", i));
    const Blob* in_i = BnInOp2Blob(ibn);
    CHECK_EQ(in_i->shape().NumAxes(), out->shape().NumAxes());
    const RangeProto& range = conf.in_split().range(i);
    CHECK_LT(range.begin(), range.end());
    CHECK_LE(range.end(), out->shape().At(in_split_axis));
    FOR_RANGE(int64_t, axis, 0, in_i->shape().NumAxes()) {
      if (axis == in_split_axis) {
        CHECK_EQ(in_i->shape().At(axis), range.end() - range.begin());
      } else {
        CHECK_EQ(in_i->shape().At(axis), out->shape().At(axis));
      }
    }
    if (in_split_axis == 0) {
      CudaCheck(cudaMemcpyAsync(out->mut_dptr<T>(range.begin()), in_i->dptr(),
                                in_i->ByteSizeOfDataContentField(), cudaMemcpyDefault,
                                ctx.device_ctx->cuda_stream()));
    } else {
      const int64_t rows = out->shape().Count(0, in_split_axis);
      const int64_t src_pitch = in_i->shape().elem_cnt() / rows * sizeof(T);
      const void* src_dptr = in_i->dptr();
      const int64_t dst_pitch = out->shape().elem_cnt() / rows * sizeof(T);
      void* dst_dptr = out->mut_dptr<char>()
                       + range.begin() * in_i->shape().Count(in_split_axis + 1) * sizeof(T);
      CudaCheck(cudaMemcpy2DAsync(dst_dptr, dst_pitch, src_dptr, src_pitch, src_pitch, rows,
                                  cudaMemcpyDefault, ctx.device_ctx->cuda_stream()));
    }
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalGpuPeerSplitToBroadcastConf,
                               LocalGpuPeerSplitToBroadcastKernel, ARITHMETIC_DATA_TYPE_SEQ)

#endif

}  // namespace oneflow
