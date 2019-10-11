#ifndef ONEFLOW_CORE_DATA_DATA_LOADER_H_
#define ONEFLOW_CORE_DATA_DATA_LOADER_H_

#include "oneflow/core/data/dataset.h"
#include "oneflow/core/data/data_sampler.h"
#include "oneflow/core/data/ring_queue.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/buffer.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include <thread>

namespace oneflow {
namespace data {

class BatchDataInstance final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchDataInstance);
  BatchDataInstance(size_t batch_size) : data_inst_vec_(batch_size), fill_count_(0) {}
  ~BatchDataInstance() = default;

  DataInstance* Get(size_t idx_in_batch) { return &(data_inst_vec_.at(idx_in_batch)); }
  size_t Size() const { return data_inst_vec_.size(); }
  void IncreaseFillCount() { fill_count_ += 1; }
  bool IsReady() { return data_inst_vec_.size() == fill_count_.load(); }
  void ForEach(std::function<void(DataInstance*)> handler) {
    for (auto& data_inst : data_inst_vec_) { handler(&data_inst); }
  }

 private:
  std::vector<DataInstance> data_inst_vec_;
  std::atomic<size_t> fill_count_;
};

class DataLoader final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoader);
  DataLoader(const DataLoadOpConf& op_conf, const DataLoadKernelConf& kernel_conf);
  ~DataLoader();

  std::shared_ptr<BatchDataInstance> FetchBatch();

 protected:
  std::shared_ptr<BatchDataInstance> AcquireGetBatch(size_t batch_size);
  void LoadBatch();
  void Close();
  bool IsImageAlignNeeded(size_t& alignment);
  void ImageAlign(std::shared_ptr<BatchDataInstance> batch_data_inst_ptr, size_t alignment);

 private:
  DataLoadOpConf op_conf_;
  DataLoadKernelConf kernel_conf_;

  std::shared_ptr<Dataset> dataset_;
  DataSamplerContext sampler_ctx_;
  Buffer<std::shared_ptr<BatchDataInstance>> batch_buffer_;

  bool is_closed_;
  std::thread load_thrd_;
  ThreadPool worker_pool_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DATA_DATA_LOADER_H_
