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
  dataset_->GetSampler()->SubmitContext(&sampler_ctx_);
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
      if (IsImageAlignNeeded(collator) && collator->IsReady()) { ImageAlign(collator); }
    });
  }
}

BatchCollator* DataLoader::GetBatchCollator() {
  auto* collator = batch_queue_.Last();
  if (!collator || collator->IsFull()) {
    batch_queue_.SyncEnqueue(std::unique_ptr<BatchCollator>(new BatchCollator(batch_size_)));
    collator = batch_queue_.Last();
  }
  return collator;
}

bool DataLoader::IsImageAlignNeeded(BatchCollator* collator) {
  return conf_->has_image_alignment()
         && collator->batch_.back()->HasField<DataSourceCase::kImage>();
}

void DataLoader::ImageAlign(BatchCollator* collator) {
  int64_t max_rows = -1;
  int64_t max_cols = -1;
  int64_t channels = -1;
  for (auto& data_inst_ptr : collator->batch_) {
    auto& image_mat =
        dynamic_cast<ImageDataField*>(data_inst_ptr->GetField<DataSourceCase::kImage>())->data();
    max_rows = std::max<int64_t>(max_rows, image_mat.rows);
    max_cols = std::max<int64_t>(max_cols, image_mat.cols);
    if (channels == -1) {
      channels = image_mat.depth();
    } else {
      CHECK_EQ(channels, image_mat.depth());
    }
  }
  CHECK_GT(max_rows, 0);
  CHECK_GT(max_cols, 0);
  CHECK_GT(channels, 0);
  max_rows = RoundUp(max_rows, conf_->image_alignment());
  max_cols = RoundUp(max_cols, conf_->image_alignment());

  for (auto& data_inst_ptr : collator->batch_) {
    auto* image_field =
        dynamic_cast<ImageDataField*>(data_inst_ptr->GetField<DataSourceCase::kImage>());
    CHECK_NOTNULL(image_field);
    worker_pool_.AddWork([image_field, max_cols, max_rows]() {
      auto& image_mat = image_field->data();
      cv::Mat dst = cv::Mat::zeros(cv::Size(max_cols, max_rows), image_mat.type());
      image_mat.copyTo(dst(cv::Rect(0, 0, image_mat.cols, image_mat.rows)));
      image_field->data() = dst;
    });
  }
}

}  // namespace data
}  // namespace oneflow
