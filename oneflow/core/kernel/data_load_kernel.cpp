#include "oneflow/core/kernel/data_load_kernel.h"
#include "oneflow/core/dataset/dataset_manager.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

void DataLoadKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  const DataLoadOpConf& data_load_op_conf = op_conf().data_load_conf();
  int64_t piece_size = Global<JobDesc>::Get()->RecordPieceSize();
  int64_t parallele_num = parallel_ctx->parallel_num();
  CHECK_EQ(piece_size % parallele_num, 0);
  int64_t device_piece_size = piece_size / parallele_num;
  size_t queue_size = Global<JobDesc>::Get()->data_load_queue_size();
  std::shared_ptr<Dataset> dataset = Global<DatasetManager>::Get()->At(data_load_op_conf.dataset());
  std::vector<int64_t> part_data_seq =
      dataset->GetPartDataSequence(parallel_ctx->parallel_id(), parallele_num);

  data_loader_.reset(
      new DataLoader(device_piece_size, queue_size, dataset, std::move(part_data_seq)));
}

void DataLoadKernel::Forward(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  data_loader_->BatchUnload(BnInOp2Blob("out")->mut_dptr<OFRecord>());
  auto* status = static_cast<DataLoadStatus*>(ctx.other);
  if (data_loader_->IsEof()) { status->is_eof = true; }
  // auto status =
  // status->record_num = record_reader_->Read(static_cast<size_t>(piece_size_in_one_loader_),
  //                                           BnInOp2Blob("out")->mut_dptr<OFRecord>());
  // BnInOp2Blob("out")->set_record_num(status->record_num);
  // if (status->record_num < piece_size_in_one_loader_) { status->is_eof = true; }
}

REGISTER_KERNEL(OperatorConf::kDataLoadConf, DataLoadKernel);

}  // namespace oneflow
