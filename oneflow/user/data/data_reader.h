/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_USER_DATA_DATA_READER_H_
#define ONEFLOW_USER_DATA_DATA_READER_H_

#include "oneflow/core/common/buffer.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/user/data/dataset.h"
#include "oneflow/user/data/parser.h"

namespace oneflow {
namespace data {

static const int32_t kDataReaderBatchBufferSize = 4;

template<typename LoadTarget>
class DataReader {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  DataReader(user_op::KernelInitContext* ctx)
      : is_closed_(false), batch_buffer_(kDataReaderBatchBufferSize) {}
  virtual ~DataReader() {
    Close();
    if (load_thrd_.joinable()) { load_thrd_.join(); }
  }

  void Read(user_op::KernelComputeContext* ctx) {
    CHECK(load_thrd_.joinable()) << "You should call StartLoadThread before read data";
    auto batch_data = FetchBatchData();
    parser_->Parse(batch_data, ctx);
  }

  void Close() {
    is_closed_.store(true);
    bool buffer_drained = false;
    while (!buffer_drained) {
      std::shared_ptr<LoadTargetPtrList> abandoned_batch_data(nullptr);
      auto status = batch_buffer_.TryReceive(&abandoned_batch_data);
      CHECK_NE(status, BufferStatus::kBufferStatusErrorClosed);
      buffer_drained = (status == BufferStatus::kBufferStatusEmpty);
    }
    batch_buffer_.Close();
  }

 protected:
  void StartLoadThread() {
    if (load_thrd_.joinable()) { return; }
    load_thrd_ = std::thread([this] {
      while (!is_closed_.load() && LoadBatch()) {}
    });
  }

  std::unique_ptr<Dataset<LoadTarget>> loader_;
  std::unique_ptr<Parser<LoadTarget>> parser_;

 private:
  std::shared_ptr<LoadTargetPtrList> FetchBatchData() {
    std::shared_ptr<LoadTargetPtrList> batch_data(nullptr);
    CHECK_EQ(batch_buffer_.Receive(&batch_data), BufferStatus::kBufferStatusSuccess);
    return batch_data;
  }

  bool LoadBatch() {
    std::shared_ptr<LoadTargetPtrList> batch_data =
        std::make_shared<LoadTargetPtrList>(std::move(loader_->Next()));
    return batch_buffer_.Send(batch_data) == BufferStatus::kBufferStatusSuccess;
  }

  std::atomic<bool> is_closed_;
  Buffer<std::shared_ptr<LoadTargetPtrList>> batch_buffer_;
  std::thread load_thrd_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_DATA_READER_H_
