#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/device/nccl_device_context.h"
#include "nccl.h"

namespace oneflow {

class NcclTupleReduceKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclTupleReduceKernel);
  NcclTupleReduceKernel() = default;
  ~NcclTupleReduceKernel() override = default;

 private:
  bool IsKernelLaunchSynchronized() const override { return false; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

void NcclTupleReduceKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NcclTupleReduceOpConf& conf = this->op_conf().nccl_tuple_reduce_conf();
  const auto& parallel_ctx = this->kernel_conf().nccl_tuple_reduce_conf().parallel_ctx();
  const int64_t num_in_out = conf.out_size();
  std::vector<const void*> send_ptr_vec(num_in_out);
  std::vector<void*> recv_ptr_vec(num_in_out);
  std::vector<int64_t> elem_cnt_vec(num_in_out);
  std::vector<DataType> data_type_vec(num_in_out);
  FOR_RANGE(int64_t, i, 0, conf.out_size()) {
    const Blob* in = BnInOp2Blob(GenRepeatedBn("in", i));
    send_ptr_vec[i] = in->dptr();
    Blob* out = BnInOp2Blob(GenRepeatedBn("out", i));
    recv_ptr_vec[i] = conf.root(i) == parallel_ctx.rank_ctx().rank_id() ? out->mut_dptr() : nullptr;
    elem_cnt_vec[i] = in->shape().elem_cnt();
    data_type_vec[i] = in->data_type();
  }
  auto* device_ctx = dynamic_cast<NcclDeviceCtx*>(ctx.device_ctx);
  CHECK_NOTNULL(device_ctx);
  device_ctx->Enqueue(
      [device_ctx, num_in_out, send_ptr_vec, recv_ptr_vec, elem_cnt_vec, data_type_vec, conf] {
        NcclCheck(ncclGroupStart());
        FOR_RANGE(int64_t, i, 0, num_in_out) {
          NcclCheck(ncclReduce(send_ptr_vec.at(i), recv_ptr_vec.at(i), elem_cnt_vec.at(i),
                               GetNcclDataType(data_type_vec.at(i)), ncclRedOp_t::ncclSum,
                               conf.root(i), device_ctx->nccl_handle(), device_ctx->cuda_stream()));
        }
        NcclCheck(ncclGroupEnd());
      });
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kNcclTupleReduceConf, DeviceType::kGPU,
                            NcclTupleReduceKernel);

}  // namespace oneflow
