#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/nccl_util.h"
#include "nccl.h"

namespace oneflow {

class NcclTupleBroadcastKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclTupleBroadcastKernel);
  NcclTupleBroadcastKernel() = default;
  ~NcclTupleBroadcastKernel() override = default;

 private:
  void VirtualKernelInit() override {}
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

namespace {

struct BlobGroup {
  int64_t root;
  char* out_ptr;
  int64_t out_size;
};

}  // namespace

void NcclTupleBroadcastKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NcclTupleBroadcastOpConf& conf = this->op_conf().nccl_tuple_broadcast_conf();
  const auto& parallel_ctx = this->kernel_conf().nccl_tuple_broadcast_conf().parallel_ctx();
  //  NcclCheck(ncclGroupStart());
  //  FOR_RANGE(int64_t, i, 0, conf.out_size()) {
  //    const void* send = conf.root(i) == parallel_ctx.rank_ctx().rank_id()
  //                           ? BnInOp2Blob(GenRepeatedBn("in", i))->dptr()
  //                           : nullptr;
  //    Blob* out = BnInOp2Blob(GenRepeatedBn("out", i));
  //    void* recv = out->mut_dptr();
  //    NcclCheck(ncclBroadcast(send, recv, out->shape().elem_cnt(),
  //    GetNcclDataType(out->data_type()),
  //                            conf.root(i), ctx.device_ctx->nccl_handle(),
  //                            ctx.device_ctx->cuda_stream()));
  //  }
  //  NcclCheck(ncclGroupEnd());
  std::vector<BlobGroup> groups;
  FOR_RANGE(int64_t, i, 0, conf.out_size()) {
    const int64_t root_i = conf.root(i);
    Blob* out_i = BnInOp2Blob(GenRepeatedBn("out", i));
    if (root_i == parallel_ctx.rank_ctx().rank_id()) {
      const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
      out_i->CopyDataContentFrom(ctx.device_ctx, in_i);
    }
    if (!groups.empty() && groups.back().root == root_i
        && groups.back().out_ptr + groups.back().out_size == out_i->dptr()) {
      groups.back().out_size += out_i->blob_desc().ByteSizeOfBlobBody();
    } else {
      groups.push_back(BlobGroup{
          .root = root_i,
          .out_ptr = out_i->mut_dptr<char>(),
          .out_size = static_cast<int64_t>(out_i->blob_desc().ByteSizeOfBlobBody()),
      });
    }
  }
  NcclCheck(ncclGroupStart());
  for (BlobGroup group : groups) {
    const void* send = group.root == parallel_ctx.rank_ctx().rank_id() ? group.out_ptr : nullptr;
    NcclCheck(ncclBroadcast(send, group.out_ptr, group.out_size, GetNcclDataType(DataType::kChar),
                            group.root, ctx.device_ctx->nccl_handle(),
                            ctx.device_ctx->cuda_stream()));
  }
  NcclCheck(ncclGroupEnd());
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kNcclTupleBroadcastConf, DeviceType::kGPU,
                            NcclTupleBroadcastKernel);

}  // namespace oneflow
