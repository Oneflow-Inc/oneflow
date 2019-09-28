#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/device/nccl_device_context.h"

namespace oneflow {

#ifdef WITH_CUDA

class NcclHierarchicalKernel : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclHierarchicalKernel);
  NcclHierarchicalKernel() = default;
  ~NcclHierarchicalKernel() override = default;
  bool IsKernelLaunchSynchronized() const override { return false; }
};

class NcclHierarchicalReduceKernel : public NcclHierarchicalKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclHierarchicalReduceKernel);
  NcclHierarchicalReduceKernel() = default;
  ~NcclHierarchicalReduceKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    const int64_t elem_cnt = out->shape().elem_cnt();
    const void* send_ptr = in->dptr();
    void* recv_ptr = out->mut_dptr();
    const DataType data_type = in->data_type();
    auto* device_ctx = dynamic_cast<NcclDeviceCtx*>(ctx.device_ctx);
    CHECK_NOTNULL(device_ctx);
    device_ctx->Enqueue([device_ctx, send_ptr, recv_ptr, elem_cnt, data_type] {
      NcclCheck(ncclReduce(send_ptr, recv_ptr, elem_cnt, GetNcclDataType(data_type), ncclSum, 0,
                           device_ctx->nccl_handle(), device_ctx->cuda_stream()));
    });
  }
};

class NcclHierarchicalAllReduceKernel : public NcclHierarchicalKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclHierarchicalAllReduceKernel);
  NcclHierarchicalAllReduceKernel() = default;
  ~NcclHierarchicalAllReduceKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    if (!kernel_conf().nccl_hierarchical_all_reduce_conf().need_do_all_reduce()) { return; }
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    const int64_t elem_cnt = in->shape().elem_cnt();
    const void* send_ptr = in->dptr();
    void* recv_ptr = out->mut_dptr();
    const DataType data_type = in->data_type();
    auto* device_ctx = dynamic_cast<NcclDeviceCtx*>(ctx.device_ctx);
    CHECK_NOTNULL(device_ctx);
    device_ctx->Enqueue([device_ctx, send_ptr, recv_ptr, elem_cnt, data_type] {
      NcclCheck(ncclAllReduce(send_ptr, recv_ptr, elem_cnt, GetNcclDataType(data_type), ncclSum,
                              device_ctx->nccl_handle(), device_ctx->cuda_stream()));
    });
  }
};

class NcclHierarchicalBroadcastKernel : public NcclHierarchicalKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclHierarchicalBroadcastKernel);
  NcclHierarchicalBroadcastKernel() = default;
  ~NcclHierarchicalBroadcastKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    const int64_t elem_cnt = in->shape().elem_cnt();
    const void* send_ptr = in->dptr();
    void* recv_ptr = out->mut_dptr();
    const DataType data_type = in->data_type();
    auto* device_ctx = dynamic_cast<NcclDeviceCtx*>(ctx.device_ctx);
    CHECK_NOTNULL(device_ctx);
    device_ctx->Enqueue([device_ctx, send_ptr, recv_ptr, elem_cnt, data_type] {
      NcclCheck(ncclBroadcast(send_ptr, recv_ptr, elem_cnt, GetNcclDataType(data_type), 0,
                              device_ctx->nccl_handle(), device_ctx->cuda_stream()));
    });
  }
};

REGISTER_KERNEL(OperatorConf::kNcclHierarchicalReduceConf, NcclHierarchicalReduceKernel);
REGISTER_KERNEL(OperatorConf::kNcclHierarchicalAllReduceConf, NcclHierarchicalAllReduceKernel);
REGISTER_KERNEL(OperatorConf::kNcclHierarchicalBroadcastConf, NcclHierarchicalBroadcastKernel);

#endif  // WITH_CUDA

}  // namespace oneflow
