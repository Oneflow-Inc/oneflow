#include "oneflow/core/kernel/ring_all_gather_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
void RingAllGatherKernel<device_type, T>::VirtualKernelInit(const ParallelContext* ctx) {
  const RingBoxingConf& conf = this->op_conf().ring_all_gather_conf().ring_boxing_conf();
  CHECK_EQ(conf.ring_size(), ctx->parallel_num());
  CHECK_EQ(conf.slices_size(), ctx->parallel_num());
  std::vector<int64_t> ring_prev(conf.ring_size());
  FOR_RANGE(int64_t, i, 0, conf.ring_size()) { ring_prev[conf.ring(i)] = i; }
  num_steps_ = conf.ring_size();
  int64_t current_slice_id = ctx->parallel_id();
  out_slice_ = TensorSliceView(Shape(conf.logical_blob_shape()));
  FOR_RANGE(int64_t, i, 0, num_steps_) {
    chunk_slices_.emplace_back(conf.slices(current_slice_id));
    current_slice_id = ring_prev[current_slice_id];
  }
}

template<DeviceType device_type, typename T>
void RingAllGatherKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t step = *static_cast<int64_t*>(ctx.other);
  if (step == 0) {
    BnInOp2Blob("send")->set_dim0_valid_num(0, BnInOp2Blob("in")->shape().elem_cnt());
  } else if (step < num_steps_) {
    BnInOp2Blob("send")->set_dim0_valid_num(0, BnInOp2Blob("recv")->dim0_valid_num(0));
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void RingAllGatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t step = *static_cast<int64_t*>(ctx.other);
  const Blob* in = BnInOp2Blob("in");
  const Blob* recv = BnInOp2Blob("recv");
  Blob* out = BnInOp2Blob("out");
  Blob* send = BnInOp2Blob("send");
  std::unique_ptr<MemoryCopier> memory_copier(NewDefaultMemoryCopier(device_type));
  const TensorSliceView chunk_slice = chunk_slices_.at(step);
  if (step == 0) {
    send->CopyValidDataContentFrom(ctx.device_ctx, in);
    TensorSliceCopier in_out_tsc(out_slice_, chunk_slice, this->kernel_conf().data_type());
    in_out_tsc.Copy(ctx.device_ctx, *memory_copier, out, in);
  } else if (step < num_steps_) {
    TensorSliceCopier recv_out_tsc(out_slice_, chunk_slice, this->kernel_conf().data_type());
    recv_out_tsc.Copy(ctx.device_ctx, *memory_copier, out, recv);
    if (step != num_steps_ - 1) { send->CopyValidDataContentFrom(ctx.device_ctx, recv); }
  } else {
    UNIMPLEMENTED();
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kRingAllGatherConf, RingAllGatherKernel, POD_DATA_TYPE_SEQ)

}  // namespace oneflow
