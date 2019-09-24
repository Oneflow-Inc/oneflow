#include "oneflow/core/data/data_loader.h"

namespace oneflow {
namespace data {

size_t BatchCollator::GetEmptySlot() {
  CHECK_LE(cur_slot_, batch_size_);
  return cur_slot_++;
}

void BatchCollator::Fill(size_t slot, std::unique_ptr<DataInstance>&& instance) {
  CHECK(!batch_.at(slot));
  batch_.at(slot).swap(instance);
}

bool BatchCollator::IsReady() const {
  for (const auto& record : batch_) {
    if (!record) { return false; }
  }
  return true;
}

void BatchCollator::ForEach(std::function<void(DataInstance*)> handler) const {
  for (const auto& record : batch_) {
    auto* raw_ptr = record.get();
    handler(raw_ptr);
  }
}

DataLoader::DataLoader(const DataLoadKernelConf& conf, std::shared_ptr<Dataset> dataset,
                       size_t qsize)
    : batch_size_(conf.device_batch_size()),
      dataset_(dataset),
      conf_(&conf),
      is_closed_(false),
      worker_pool_(std::thread::hardware_concurrency() / 4),
      batch_queue_(qsize) {
  sampler_ctx_.num_replicas_ = conf.parallel_num();
  sampler_ctx_.rank_ = conf.parallel_id();
  sampler_ctx_.epoch_ = 0;
  sampler_ctx_.iter_ = conf.parallel_id();
  sampler_ctx_.count_ = 0;
  load_thrd_ = std::thread([this] {
    while (!is_closed_) { LoadBatch(); }
  });
}

DataLoader::~DataLoader() {
  Close();
  load_thrd_.join();
}

std::vector<std::unique_ptr<DataInstance>> DataLoader::FetchBatch() {
  std::unique_ptr<BatchCollator> collator =
      batch_queue_.SyncDequeue([](const BatchCollator* s) { return s->IsReady(); });
  return std::move(collator->batch_);
}

void DataLoader::LoadBatch() {
  std::vector<int64_t> batch_idx_seq =
      dataset_->GetSampler()->FetchBatchIndexSequence(&sampler_ctx_, batch_size_);
  for (int64_t idx : batch_idx_seq) {
    auto* collator = GetBatchCollator();
    size_t slot = collator->GetEmptySlot();
    worker_pool_.AddWork([this, collator, slot, idx]() {
      auto data_inst = std::make_unique<DataInstance>(conf_->data_instance());
      dataset_->GetData(idx, data_inst.get());
      for (const auto& trans_proto : conf_->transforms()) { data_inst->Transform(trans_proto); }
      collator->Fill(slot, std::move(data_inst));
    });
  }
}

BatchCollator* DataLoader::GetBatchCollator() {
  auto* collator = batch_queue_.Tail();
  if (collator->IsFull()) {
    batch_queue_.SyncEnqueue(std::unique_ptr<BatchCollator>(new BatchCollator(batch_size_)));
    collator = batch_queue_.Tail();
  }
  return collator;
}

}  // namespace data
}  // namespace oneflow
