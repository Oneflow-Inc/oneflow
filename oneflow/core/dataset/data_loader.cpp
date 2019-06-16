#include "oneflow/core/dataset/data_loader.h"
#include "oneflow/core/dataset/dataset.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

size_t BatchSampler::GetEmptySlot() {
  CHECK_LE(cur_slot_, batch_size_);
  return cur_slot_++;
}

void BatchSampler::Fill(size_t slot, std::unique_ptr<OFRecord>&& record) {
  CHECK(!batch_.at(slot));
  batch_.at(slot).swap(record);
}

bool BatchSampler::IsReady() const {
  for (const auto& record : batch_) {
    if (!record) { return false; }
  }
  return true;
}

void BatchSampler::ForEach(std::function<void(const OFRecord*)> handler) const {
  for (const auto& record : batch_) {
    const auto* raw_ptr = record.get();
    handler(raw_ptr);
  }
}

DataLoader::DataLoader(size_t batch_size, size_t qsize, std::shared_ptr<Dataset> dataset,
                       std::vector<int64_t>&& data_seq)
    : dataset_(dataset),
      data_seq_(data_seq),
      cur_load_data_idx_(0),
      batch_size_(batch_size),
      worker_pool_(std::thread::hardware_concurrency() / 4),
      batch_queue_(qsize) {
  dispatcher_ = std::thread([this] {
    while (Dispatch()) {}
  });
}

void DataLoader::BatchUnload(OFRecord* record_array) {
  std::unique_ptr<BatchSampler> sampler =
      batch_queue_.SyncDequeue([](const BatchSampler* s) { return s->IsReady(); });
  size_t idx = 0;
  sampler->ForEach([record_array, &idx](const OFRecord* record) {
    record_array[idx] = *record;
    ++idx;
  });
}

size_t DataLoader::Dispatch() {
  auto* sampler = GetBatchSampler();
  size_t slot = sampler->GetEmptySlot();
  int64_t data_idx = data_seq_.at(cur_load_data_idx_);
  worker_pool_.AddWork([this, sampler, slot, data_idx]() {
    std::unique_ptr<OFRecord> record = dataset_->EncodeOneRecord(data_idx);
    sampler->Fill(slot, std::move(record));
  });
  ++cur_load_data_idx_;
  return data_seq_.size() - cur_load_data_idx_;
}

BatchSampler* DataLoader::GetBatchSampler() {
  auto* sampler = batch_queue_.Tail();
  if (sampler->IsFull()) {
    batch_queue_.SyncEnqueue(std::unique_ptr<BatchSampler>(new BatchSampler(batch_size_)));
    sampler = batch_queue_.Tail();
  }
  return sampler;
}

}  // namespace oneflow
