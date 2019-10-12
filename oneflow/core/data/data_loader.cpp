#include "oneflow/core/data/data_loader.h"
#include "oneflow/core/data/dataset_manager.h"
#include "oneflow/core/common/blocking_counter.h"

namespace oneflow {
namespace data {

DataLoader::DataLoader(const DataLoadOpConf& op_conf, const DataLoadKernelConf& kernel_conf)
    : op_conf_(op_conf),
      kernel_conf_(kernel_conf),
      batch_buffer_(op_conf.batch_cache_size()),
      is_closed_(false),
      worker_pool_(std::thread::hardware_concurrency() / 4) {
  dataset_ = Global<DatasetManager>::Get()->GetOrCreateDataset(op_conf.dataset());
  sampler_ctx_.num_replicas_ = kernel_conf.parallel_num();
  sampler_ctx_.rank_ = kernel_conf.parallel_id();
  sampler_ctx_.epoch_ = 0;
  sampler_ctx_.iter_ = kernel_conf.parallel_id();
  sampler_ctx_.count_ = 0;
  dataset_->SubmitSamplerContext(&sampler_ctx_);
  load_thrd_ = std::thread([this] {
    while (!is_closed_) { LoadBatch(); }
  });
}

DataLoader::~DataLoader() {
  Close();
  load_thrd_.join();
}

void DataLoader::Close() {
  is_closed_ = true;
  batch_buffer_.Close();
}

std::shared_ptr<DataLoader::BatchDataInstance> DataLoader::FetchBatch() {
  std::shared_ptr<BatchDataInstance> batch_data_inst_ptr(nullptr);
  batch_buffer_.Receive(&batch_data_inst_ptr);
  return batch_data_inst_ptr;
}

void DataLoader::LoadBatch() {
  std::vector<int64_t> batch_idx_seq = dataset_->FetchBatchIndexSequence(
      &sampler_ctx_, kernel_conf_.device_batch_size());

  BatchDataInstance batch_data(batch_idx_seq.size());
  auto batch_data_inst_ptr = std::make_shared<BatchDataInstance>(std::move(batch_data));

  BlockingCounter bc(batch_idx_seq.size());
  FOR_RANGE(size_t, idx_in_batch, 0, batch_idx_seq.size()) {
    int64_t data_idx = batch_idx_seq.at(idx_in_batch);
    worker_pool_.AddWork([this, data_idx, idx_in_batch, batch_data_inst_ptr]() {
      DataInstance* data_inst = &(batch_data_inst_ptr->at(idx_in_batch));
      data_inst->InitFromProto(kernel_conf_.data_instance());
      dataset_->GetData(data_idx, data_inst);
      for (const auto& trans_proto : op_conf_.transforms()) { data_inst->Transform(trans_proto); }
    });
  }
  bc.WaitUntilCntEqualZero();

  // TODO: implement ImageAlign with batch transform
  size_t image_alignment = 1;
  if (IsImageAlignNeeded(image_alignment)) {
    ImageAlign(batch_data_inst_ptr.get(), image_alignment);
  }
  batch_buffer_.Send(batch_data_inst_ptr);
}

bool DataLoader::IsImageAlignNeeded(size_t& alignment) {
  for (const auto& trans_proto : op_conf_.transforms()) {
    if (trans_proto.has_target_resize()) {
      alignment = trans_proto.target_resize().alignment();
      return true;
    }
  }
  return false;
}

void DataLoader::ImageAlign(BatchDataInstance* batch_data_inst,
                            size_t alignment) {
  int64_t max_rows = -1;
  int64_t max_cols = -1;
  int64_t channels = -1;
  bool has_image_field = true;

  for (DataInstance& data_inst : *batch_data_inst) {
    auto* image_field =
        dynamic_cast<ImageDataField*>(data_inst.GetField<DataSourceCase::kImage>());
    if (image_field == nullptr) {
      has_image_field = false;
      break;
    }
    auto& image_mat = image_field->data();
    max_rows = std::max<int64_t>(max_rows, image_mat.rows);
    max_cols = std::max<int64_t>(max_cols, image_mat.cols);
    if (channels == -1) {
      channels = image_mat.channels();
    } else {
      CHECK_EQ(channels, image_mat.channels());
    }
  }
  if (!has_image_field) { return; }

  CHECK_GT(max_rows, 0);
  CHECK_GT(max_cols, 0);
  CHECK_GT(channels, 0);
  max_rows = RoundUp(max_rows, alignment);
  max_cols = RoundUp(max_cols, alignment);

  BlockingCounter bc(batch_data_inst->size());
  for (DataInstance& data_inst : *batch_data_inst) {
    auto* image_field =
        dynamic_cast<ImageDataField*>(data_inst.GetField<DataSourceCase::kImage>());
    CHECK_NOTNULL(image_field);
    worker_pool_.AddWork([=]() {
      auto& image_mat = image_field->data();
      cv::Mat dst = cv::Mat::zeros(cv::Size(max_cols, max_rows), image_mat.type());
      image_mat.copyTo(dst(cv::Rect(0, 0, image_mat.cols, image_mat.rows)));
      image_field->data() = dst;
    });
  }
  bc.WaitUntilCntEqualZero();
}

}  // namespace data
}  // namespace oneflow
