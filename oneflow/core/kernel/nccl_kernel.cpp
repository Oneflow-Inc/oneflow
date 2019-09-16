#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/device/nccl_device_context.h"

namespace oneflow {

#ifdef WITH_CUDA

class NcclReduceScatterKernel : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclReduceScatterKernel);
  NcclReduceScatterKernel() = default;
  ~NcclReduceScatterKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    const int64_t elem_cnt = out->shape().elem_cnt();
    const void* send = in->dptr();
    void* recv = out->mut_dptr();
    const DataType data_type = in->data_type();
    auto* device_ctx = dynamic_cast<NcclDeviceCtx*>(ctx.device_ctx);
    CHECK_NOTNULL(device_ctx);
    device_ctx->Enqueue([device_ctx, send, recv, elem_cnt, data_type] {
      NcclCheck(ncclReduceScatter(send, recv, elem_cnt, GetNcclDataType(data_type), ncclSum,
                                  device_ctx->nccl_handle(), device_ctx->cuda_stream()));
    });
  }
};

class NcclAllGatherKernel : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclAllGatherKernel);
  NcclAllGatherKernel() = default;
  ~NcclAllGatherKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    const int64_t elem_cnt = in->shape().elem_cnt();
    const void* send = in->dptr();
    void* recv = out->mut_dptr();
    const DataType data_type = in->data_type();
    auto* device_ctx = dynamic_cast<NcclDeviceCtx*>(ctx.device_ctx);
    CHECK_NOTNULL(device_ctx);
    device_ctx->Enqueue([device_ctx, send, recv, elem_cnt, data_type] {
      NcclCheck(ncclAllGather(send, recv, elem_cnt, GetNcclDataType(data_type),
                              device_ctx->nccl_handle(), device_ctx->cuda_stream()));
    });
  }
};

class NcclAllReduceKernel : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclAllReduceKernel);
  NcclAllReduceKernel() = default;
  ~NcclAllReduceKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    const int64_t elem_cnt = in->shape().elem_cnt();
    const void* send = in->dptr();
    void* recv = out->mut_dptr();
    const DataType data_type = in->data_type();
    auto* device_ctx = dynamic_cast<NcclDeviceCtx*>(ctx.device_ctx);
    CHECK_NOTNULL(device_ctx);
    device_ctx->Enqueue([device_ctx, send, recv, elem_cnt, data_type] {
      NcclCheck(ncclAllReduce(send, recv, elem_cnt, GetNcclDataType(data_type), ncclSum,
                              device_ctx->nccl_handle(), device_ctx->cuda_stream()));
    });
  }
};

REGISTER_KERNEL(OperatorConf::kNcclReduceScatterConf, NcclReduceScatterKernel);
REGISTER_KERNEL(OperatorConf::kNcclAllGatherConf, NcclAllGatherKernel);
REGISTER_KERNEL(OperatorConf::kNcclAllReduceConf, NcclAllReduceKernel);

#endif  // WITH_CUDA

}  // namespace oneflow
