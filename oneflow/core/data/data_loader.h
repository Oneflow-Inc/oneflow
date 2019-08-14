#ifndef ONEFLOW_CORE_DATASET_DATA_LOADER_H_
#define ONEFLOW_CORE_DATASET_DATA_LOADER_H_

#include "oneflow/core/data/dataset.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/data/ring_queue.h"
#include <thread>

namespace oneflow {
namespace data {

class BatchCollator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchCollator);
  BatchCollator() = delete;
  ~BatchCollator() = default;
  BatchCollator(size_t batch_size) : batch_size_(batch_size), cur_slot_(0), batch_(batch_size) {}

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
  DataLoader(std::shared_ptr<Dataset> dataset, size_t batch_size, size_t max_count, size_t qsize,
             size_t num_replicas, size_t rank);

  bool IsEof() const { return load_count_ == 0; }
  void DumpToRecrod(OFRecord* record_array);

 private:
  size_t LoadBatch();
  BatchCollator* GetBatchCollator();

 private:
  size_t batch_size_;
  size_t load_count_;
  std::shared_ptr<Dataset> dataset_;
  DataSamplerContext sampler_ctx_;

  std::thread load_thrd_;
  ThreadPool worker_pool_;
  util::RingQueue<BatchCollator> batch_queue_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATASET_DATA_LOADER_H_
