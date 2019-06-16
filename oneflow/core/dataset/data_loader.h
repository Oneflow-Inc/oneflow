#ifndef ONEFLOW_CORE_DATASET_DATA_LOADER_H_
#define ONEFLOW_CORE_DATASET_DATA_LOADER_H_

#include "oneflow/core/dataset/dataset.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/dataset/ring_queue.h"
#include <thread>

namespace oneflow {

class BatchSampler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchSampler);
  BatchSampler() = delete;
  ~BatchSampler() = default;
  BatchSampler(size_t batch_size) : batch_size_(batch_size), cur_slot_(0), batch_(batch_size) {}

  size_t GetEmptySlot();
  bool IsFull() const { return cur_slot_ == batch_size_; }
  bool IsReady() const;
  void Fill(size_t slot, std::unique_ptr<OFRecord>&& record);
  void ForEach(std::function<void(const OFRecord*)>) const;

 private:
  size_t batch_size_;
  size_t cur_slot_;
  std::vector<std::unique_ptr<OFRecord>> batch_;
};

class DataLoader final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoader);
  DataLoader() = delete;
  ~DataLoader() = default;
  DataLoader(size_t batch_size, size_t qsize, std::shared_ptr<Dataset> dataset,
             std::vector<int64_t>&& data_seq);

  bool IsEof() const { return cur_load_data_idx_ == data_seq_.size(); }
  void BatchUnload(OFRecord* record_array);

 private:
  size_t Dispatch();
  BatchSampler* GetBatchSampler();

 private:
  std::shared_ptr<Dataset> dataset_;
  std::vector<int64_t> data_seq_;
  size_t cur_load_data_idx_;
  size_t batch_size_;

  std::thread dispatcher_;
  ThreadPool worker_pool_;
  RingQueue<BatchSampler> batch_queue_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATASET_DATA_LOADER_H_
