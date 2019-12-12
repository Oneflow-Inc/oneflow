#include "oneflow/core/data/data_loader.h"
#include "oneflow/core/data/dataset_manager.h"
#include "oneflow/core/data/data_transform.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {
namespace data {

DataLoader::DataLoader(const DataLoadOpConf& op_conf, const DataLoadKernelConf& kernel_conf)
    : op_conf_(op_conf),
      kernel_conf_(kernel_conf),
      batch_buffer_(op_conf.batch_cache_size()),
      is_closed_(false) {
  dataset_ = Global<DatasetManager>::Get()->GetOrCreateDataset(op_conf.dataset());
  sampler_ctx_.num_replicas_ = kernel_conf.parallel_num();
  sampler_ctx_.rank_ = kernel_conf.parallel_id();
  sampler_ctx_.epoch_ = 0;
  sampler_ctx_.offset_ = kernel_conf.parallel_id();
  sampler_ctx_.count_ = 0;
  dataset_->SubmitSamplerContext(&sampler_ctx_);
  load_thrd_ = std::thread([this] {
    while (!is_closed_.load()) { LoadBatch(); }
  });
}

DataLoader::~DataLoader() {
  Close();
  load_thrd_.join();
}

void DataLoader::Close() {
  is_closed_.store(true);
  bool buffer_drained = false;
  while (!buffer_drained) {
    std::shared_ptr<BatchDataInstance> abandoned_batch_data(nullptr);
    auto status = batch_buffer_.TryReceive(&abandoned_batch_data);
    CHECK_NE(status, BufferStatus::kBufferStatusErrorClosed);
    buffer_drained = (status == BufferStatus::kBufferStatusEmpty);
  }
  batch_buffer_.Close();
}

std::shared_ptr<DataLoader::BatchDataInstance> DataLoader::FetchBatch() {
  std::shared_ptr<BatchDataInstance> batch_data_inst_ptr(nullptr);
  batch_buffer_.Receive(&batch_data_inst_ptr);
  return batch_data_inst_ptr;
}

void DataLoader::LoadBatch() {
  std::vector<int64_t> batch_idx_seq =
      dataset_->FetchBatchIndexSequence(&sampler_ctx_, kernel_conf_.device_batch_size());

  BatchDataInstance batch_data(batch_idx_seq.size());
  auto batch_data_inst_ptr = std::make_shared<BatchDataInstance>(std::move(batch_data));

  MultiThreadLoop(batch_idx_seq.size(),
                  [this, batch_data_inst_ptr, &batch_idx_seq](size_t idx_in_batch) {
                    DataInstance* data_inst = &(batch_data_inst_ptr->at(idx_in_batch));
                    data_inst->InitFromProto(kernel_conf_.data_instance());
                    int64_t data_idx = batch_idx_seq.at(idx_in_batch);
                    dataset_->GetData(data_idx, data_inst);
                  });

  for (const auto& trans_proto : op_conf_.transforms()) {
    BatchTransform(batch_data_inst_ptr, trans_proto);
  }
  batch_buffer_.Send(batch_data_inst_ptr);
}

}  // namespace data
}  // namespace oneflow
