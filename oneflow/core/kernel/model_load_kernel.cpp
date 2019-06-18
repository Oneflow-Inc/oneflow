#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/kernel/model_load_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ModelLoadKernel<device_type, T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  std::tie(part_id_, part_num_) = GetPartIdAndPartNumFromParallelCtx(parallel_ctx);
}

template<DeviceType device_type, typename T>
void ModelLoadKernel<device_type, T>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // Extract snapshot path for loadding from job IOConf
  const std::string &snapshot_path = Global<const IOConf>::Get()->model_load_snapshot_path();
  // Deserialize LogicalBlobId from string to get op_name and blob_name
  const LogicalBlobId lbi = GenLogicalBlobId(this->op_conf().model_load_conf().lbn());
  // Construct full path by snapshot path and op name
  std::string load_path = JoinPath(snapshot_path, lbi.op_name());
  std::string blob_name = lbi.blob_name();
  // Output blob which will be loaded
  Blob* out_blob = BnInOp2Blob("out");
  // Load from directory
  KernelUtil<device_type, T>::InitializeWithDir(ctx.device_ctx, part_id_, part_num_, load_path,
                                                out_blob, blob_name, out_blob->shape().At(0),
                                                out_blob->shape().Count(1));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kModelLoadConf, ModelLoadKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
