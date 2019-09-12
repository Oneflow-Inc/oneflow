#include "oneflow/core/data/data_loader.h"
#include "oneflow/core/data/dataset.h"
#include "oneflow/core/data/data_sampler.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace data {

size_t BatchCollator::GetEmptySlot() {
  CHECK_LE(cur_slot_, batch_size_);
  return cur_slot_++;
}

void BatchCollator::Fill(size_t slot, std::unique_ptr<OFRecord>&& record) {
  CHECK(!batch_.at(slot));
  batch_.at(slot).swap(record);
}

bool BatchCollator::IsReady() const {
  for (const auto& record : batch_) {
    if (!record) { return false; }
  }
  return true;
}

void BatchCollator::ForEach(std::function<void(const OFRecord*)> handler) const {
  for (const auto& record : batch_) {
    const auto* raw_ptr = record.get();
    handler(raw_ptr);
  }
}

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, 
                       size_t qsize, size_t num_replicas, size_t rank)
    : batch_size_(batch_size),
      dataset_(dataset),
      is_closed_(false),
      worker_pool_(std::thread::hardware_concurrency() / 4),
      batch_queue_(qsize) {
  sampler_ctx_.num_replicas_ = num_replicas;
  sampler_ctx_.rank_ = rank;
  sampler_ctx_.epoch_ = 0;
  sampler_ctx_.iter_ = rank;
  sampler_ctx_.count_ = 0;
  load_thrd_ = std::thread([this] {
    while (!is_closed_) {
      LoadBatch();
    }
  });
}

DataLoader::~DataLoader() {
  Close();
  load_thrd_.join();
}

void DataLoader::DumpToRecrod(OFRecord* record_array) {
  std::unique_ptr<BatchCollator> collator =
      batch_queue_.SyncDequeue([](const BatchCollator* s) { return s->IsReady(); });
  size_t idx = 0;
  collator->ForEach([record_array, &idx](const OFRecord* record) {
    record_array[idx] = *record;
    ++idx;
  });
}

void DataLoader::LoadBatch() {
  std::vector<int64_t> batch_idx_seq = dataset_->GetSampler()->FetchBatchIndexSequence(
      &sampler_ctx_, batch_size_);
  for (int64_t idx : batch_idx_seq) {
    auto* collator = GetBatchCollator();
    size_t slot = collator->GetEmptySlot();
    worker_pool_.AddWork([this, collator, slot, idx]() {
      std::unique_ptr<OFRecord> record = dataset_->EncodeOneRecord(idx);
      collator->Fill(slot, std::move(record));
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
