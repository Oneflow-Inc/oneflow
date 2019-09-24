#ifndef ONEFLOW_CORE_DATA_DATA_LOADER_H_
#define ONEFLOW_CORE_DATA_DATA_LOADER_H_

#include "oneflow/core/data/dataset.h"
#include "oneflow/core/data/data_sampler.h"
#include "oneflow/core/data/ring_queue.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/kernel/kernel.pb.h"
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
  void Fill(size_t slot, std::unique_ptr<DataInstance>&& data_inst);
  void ForEach(std::function<void(DataInstance*)>) const;

 private:
  friend class DataLoader;
  size_t batch_size_;
  size_t cur_slot_;
  std::vector<std::unique_ptr<DataInstance>> batch_;
};

class DataLoader final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoader);
  DataLoader() = delete;
  ~DataLoader();
  DataLoader(const DataLoadKernelConf& conf, std::shared_ptr<Dataset> dataset, size_t qsize);

  void Close() { is_closed_ = true; }
  std::vector<std::unique_ptr<DataInstance>> FetchBatch();

 private:
  void LoadBatch();
  BatchCollator* GetBatchCollator();

 private:
  size_t batch_size_;
  std::shared_ptr<Dataset> dataset_;
  DataSamplerContext sampler_ctx_;
  const DataLoadKernelConf* conf_;

  bool is_closed_;
  std::thread load_thrd_;
  ThreadPool worker_pool_;
  util::RingQueue<BatchCollator> batch_queue_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATA_DATA_LOADER_H_
