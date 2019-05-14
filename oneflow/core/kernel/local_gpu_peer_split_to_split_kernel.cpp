#include "oneflow/core/kernel/local_gpu_peer_split_to_split_kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

template<typename T>
void LocalGpuPeerSplitToSplitKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out = BnInOp2Blob("out");
  const int64_t in_split_axis = op_conf().local_gpu_peer_split_to_split_conf().in_split_axis();
  const int64_t out_split_axis = op_conf().local_gpu_peer_split_to_split_conf().out_split_axis();
  cudaMemcpy3DParms param = {};
  param.kind = cudaMemcpyDefault;
  const int64_t first_axis = std::min(in_split_axis, out_split_axis);
  const int64_t second_axis = std::max(in_split_axis, out_split_axis);
  const int64_t first_inner_size = out->shape().Count(first_axis + 1, second_axis);
  const int64_t second_inner_size = out->shape().Count(second_axis + 1);
  param.dstPtr = make_cudaPitchedPtr(out->mut_dptr(), out->shape().Count(second_axis) * sizeof(T),
                                     out->shape().Count(second_axis) * sizeof(T),
                                     out->shape().Count(first_axis, second_axis));
  const LocalGpuPeerSplitToSplitKernelConf& conf =
      kernel_conf().local_gpu_peer_split_to_split_conf();
  const RangeProto& out_range = conf.out_range();
  CHECK_GE(out_range.begin(), 0);
  CHECK_LT(out_range.begin(), out_range.end());
  FOR_RANGE(int32_t, i, 0, op_attribute().input_bns_size()) {
    const std::string& ibn = op_attribute().input_bns(i);
    CHECK_EQ(ibn, GenRepeatedBn("in", i));
    Blob* in_i = BnInOp2Blob(ibn);
    CHECK_EQ(in_i->shape().NumAxes(), out->shape().NumAxes());
    const RangeProto& in_range = conf.in_split().range(i);
    CHECK_GE(in_range.begin(), 0);
    CHECK_LT(in_range.begin(), in_range.end());
    CHECK_LE(in_range.end(), out->shape().At(in_split_axis));
    FOR_RANGE(int64_t, axis, 0, in_i->shape().NumAxes()) {
      if (axis == in_split_axis) {
        CHECK_EQ(in_i->shape().At(axis), in_range.end() - in_range.begin());
      } else if (axis == out_split_axis) {
        CHECK_LE(out_range.end(), in_i->shape().At(axis));
      } else {
        CHECK_EQ(in_i->shape().At(axis), out->shape().At(axis));
      }
    }
    param.srcPtr = make_cudaPitchedPtr(
        in_i->mut_dptr(), in_i->shape().Count(second_axis) * sizeof(T),
        in_i->shape().Count(second_axis) * sizeof(T), in_i->shape().Count(first_axis, second_axis));
    if (in_split_axis == first_axis && out_split_axis == second_axis) {
      param.dstPos = make_cudaPos(0, in_range.begin() * first_inner_size, 0);
      param.srcPos = make_cudaPos(out_range.begin() * second_inner_size * sizeof(T), 0, 0);
      param.extent = make_cudaExtent(
          out->shape().At(out_split_axis) * second_inner_size * sizeof(T),
          in_i->shape().At(in_split_axis) * first_inner_size, out->shape().Count(0, first_axis));
    } else if (in_split_axis == second_axis && out_split_axis == first_axis) {
      param.dstPos = make_cudaPos(in_range.begin() * second_inner_size * sizeof(T), 0, 0);
      param.srcPos = make_cudaPos(0, out_range.begin() * first_inner_size, 0);
      param.extent = make_cudaExtent(
          in_i->shape().At(in_split_axis) * second_inner_size * sizeof(T),
          out->shape().At(out_split_axis) * first_inner_size, out->shape().Count(0, first_axis));
    } else {
      UNIMPLEMENTED();
    }
    CudaCheck(cudaMemcpy3DAsync(&param, ctx.device_ctx->cuda_stream()));
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalGpuPeerSplitToSplitConf,
                               LocalGpuPeerSplitToSplitKernel, ARITHMETIC_DATA_TYPE_SEQ)

#endif

}  // namespace oneflow
