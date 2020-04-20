#ifndef ONEFLOW_CUSTOMIZED_DATA_DATA_READER_H_
#define ONEFLOW_CUSTOMIZED_DATA_DATA_READER_H_

#include "oneflow/core/common/buffer.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/customized/data/dataset.h"
#include "oneflow/customized/data/parser.h"

namespace oneflow {

template<typename LoadTarget>
class DataReader {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using BatchLoadTargetPtr = std::vector<LoadTargetPtr>;
  DataReader(user_op::KernelInitContext* ctx) : is_closed_(false), batch_buffer_(4) {
    batch_size_ = ctx->GetAttr<int32_t>("batch_size");
    load_thrd_ = std::thread([this] {
      while (!is_closed_.load()) { LoadBatch(); }
    });
  }
  ~DataReader() {
    Close();
    load_thrd_.join();
  }

  void Read(user_op::KernelComputeContext* ctx) {
    auto batch_data = FetchBatchData();
    parser_->Parse(batch_data, ctx);
  }

  void Close() {
    is_closed_.store(true);
    bool buffer_drained = false;
    while (!buffer_drained) {
      std::shared_ptr<BatchLoadTargetPtr> abandoned_batch_data(nullptr);
      auto status = batch_buffer_.TryReceive(&abandoned_batch_data);
      CHECK_NE(status, BufferStatus::kBufferStatusErrorClosed);
      buffer_drained = (status == BufferStatus::kBufferStatusEmpty);
    }
    batch_buffer_.Close();
  }

 protected:
  std::unique_ptr<Dataset<LoadTarget>> loader_;
  std::unique_ptr<Parser<LoadTarget>> parser_;

 private:
  std::shared_ptr<BatchLoadTargetPtr> FetchBatchData() {
    std::shared_ptr<BatchLoadTargetPtr> batch_data(nullptr);
    batch_buffer_.Receive(&batch_data);
    return batch_data;
  }

  void LoadBatch() {
    std::shared_ptr<BatchLoadTargetPtr> batch_data =
        std::make_shared<BatchLoadTargetPtr>(loader_->LoadBatch(batch_size_));
    batch_buffer_.Send(batch_data);
  }

  std::atomic<bool> is_closed_;
  Buffer<std::shared_ptr<BatchLoadTargetPtr>> batch_buffer_;
  std::thread load_thrd_;
  int64_t batch_size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_DATA_READER_H_
