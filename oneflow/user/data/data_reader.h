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

#include "oneflow/user/data/dataset.h"
#include "oneflow/user/data/parser.h"
#include "oneflow/core/common/buffer.h"

namespace oneflow {

namespace data {

static const int32_t kDataReaderBatchBufferSize = 4;

template<typename LoadTarget>
class DataReader {
 public:
  using SampleType = LoadTarget;
  using BatchType = std::vector<SampleType>;

  DataReader(user_op::KernelInitContext* ctx)
      : is_closed_(false), batch_buffer_(kDataReaderBatchBufferSize) {}

  virtual ~DataReader() {
    Close();
    if (load_thrd_.joinable()) { load_thrd_.join(); }
  }

  void Read(user_op::KernelComputeContext* ctx) {
    CHECK(load_thrd_.joinable()) << "You should call StartLoadThread before read data";
    auto batch = FetchBatchData();
    parser_->Parse(batch, ctx);
  }

  void Close() {
    if (!is_closed_.load()) {
      is_closed_.store(true);
      batch_buffer_.Close();
    }
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
  BatchType FetchBatchData() {
    BatchType batch;
    CHECK_EQ(batch_buffer_.Pull(&batch), BufferStatus::kBufferStatusSuccess);
    return batch;
  }

  bool LoadBatch() {
    BatchType batch = loader_->Next();
    return batch_buffer_.Push(std::move(batch)) == BufferStatus::kBufferStatusSuccess;
  }

  std::atomic<bool> is_closed_;
  Buffer<BatchType> batch_buffer_;
  std::thread load_thrd_;
};

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_DATA_READER_H_
