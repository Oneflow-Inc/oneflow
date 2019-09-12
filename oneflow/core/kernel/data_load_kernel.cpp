#include "oneflow/core/kernel/data_load_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void DataLoadKernel<device_type>::VirtualKernelInit() {
  using namespace data;
  const DataLoadOpConf& op_conf = op_conf().data_load_conf();
  int64_t piece_size = GlobalJobDesc().job_conf().piece_size();
  int64_t parallel_num = parallel_ctx->parallel_num();
  CHECK_EQ(piece_size % parallel_num, 0);
  int64_t device_piece_size = piece_size / parallel_num;
  size_t queue_size = op_conf->data_load_queue_size();
  std::shared_ptr<Dataset> dataset = Global<DatasetManager>::Get()->GetOrCreateDataset(op_conf.dataset());
  data_loader_.reset(new DataLoader(dataset, device_piece_size, queue_size,
                                    parallel_ctx->parallel_num(), parallel_ctx->parallel_id()));
}

template<DeviceType device_type>
void DataLoadKernel<device_type>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDataLoadConf, DataLoadKernel);

}  // namespace oneflow
