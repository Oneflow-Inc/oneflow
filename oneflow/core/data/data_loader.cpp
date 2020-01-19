#include "oneflow/core/data/data_loader.h"
#include "oneflow/core/data/dataset_manager.h"
#include "oneflow/core/data/data_transform.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/nvtx3/nvToolsExt.h"

namespace oneflow {
namespace data {

DataLoader::DataLoader(const DataLoadOpConf& op_conf, const DataLoadKernelConf& kernel_conf)
    : op_conf_(op_conf),
      kernel_conf_(kernel_conf),
      batch_buffer_(op_conf.num_parallels() * 2),
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
  const std::string mark("DataLoader::LoadBatch");
  nvtxRangePush(mark.c_str());
  const size_t batch_size = kernel_conf_.device_batch_size();
  const size_t num_batchs = op_conf_.num_parallels();
  std::vector<int64_t> idx_vec;
  idx_vec.reserve(num_batchs * batch_size);
  FOR_RANGE(size_t, i, 0, num_batchs) {
    std::vector<int64_t> batch_idx_vec =
        dataset_->FetchBatchIndexSequence(&sampler_ctx_, batch_size);
    idx_vec.insert(idx_vec.end(), batch_idx_vec.begin(), batch_idx_vec.end());
  }
  std::vector<std::shared_ptr<BatchDataInstance>> batch_data_inst_vec(num_batchs);
  MultiThreadLoop(num_batchs, [this, batch_size, &batch_data_inst_vec, &idx_vec](size_t i) {
    BatchDataInstance batch_data(batch_size);
    batch_data_inst_vec[i] = std::make_shared<BatchDataInstance>(std::move(batch_data));
    FOR_RANGE(size_t, j, 0, batch_size) {
      DataInstance* data_inst = &(batch_data_inst_vec[i]->at(j));
      data_inst->InitFromProto(kernel_conf_.data_instance());
      int64_t data_idx = idx_vec.at(i * batch_size + j);
      dataset_->GetData(data_idx, data_inst);
    }
    for (const auto& trans_proto : op_conf_.transforms()) {
      BatchTransform(batch_data_inst_vec[i], trans_proto);
    }
  });
  for (auto batch_data_inst_ptr : batch_data_inst_vec) { batch_buffer_.Send(batch_data_inst_ptr); }
  nvtxRangePop();
}

}  // namespace data
}  // namespace oneflow
