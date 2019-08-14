#include "oneflow/core/kernel/data_load_kernel.h"
#include "oneflow/core/data/dataset_manager.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

void DataLoadKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  using namespace data;
  const DataLoadOpConf& data_load_op_conf = op_conf().data_load_conf();
  int64_t piece_size = Global<JobDesc>::Get()->RecordPieceSize();
  int64_t parallel_num = parallel_ctx->parallel_num();
  CHECK_EQ(piece_size % parallel_num, 0);
  int64_t device_piece_size = piece_size / parallel_num;
  size_t queue_size = Global<JobDesc>::Get()->data_load_queue_size();
  std::shared_ptr<Dataset> dataset = Global<DatasetManager>::Get()->At(data_load_op_conf.dataset());
  int64_t total_num_data =
      Global<JobDesc>::Get()->IsTrain()
          ? Global<JobDesc>::Get()->TotalBatchNum() * Global<JobDesc>::Get()->BatchSize()
          : dataset->Size();
  CHECK_LE(parallel_num, total_num_data);
  BalancedSplitter bs(total_num_data, parallel_num);
  Range range = bs.At(parallel_ctx->parallel_id());
  data_loader_.reset(new DataLoader(dataset, device_piece_size, range.size(), queue_size,
                                    parallel_ctx->parallel_num(), parallel_ctx->parallel_id()));
}

void DataLoadKernel::Forward(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  data_loader_->DumpToRecrod(BnInOp2Blob("out")->mut_dptr<OFRecord>());
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
