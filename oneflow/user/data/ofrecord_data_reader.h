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
#ifndef ONEFLOW_USER_DATA_OFRECORD_DATA_READER_H_
#define ONEFLOW_USER_DATA_OFRECORD_DATA_READER_H_

#include "oneflow/user/data/data_reader.h"
#include "oneflow/user/data/ofrecord_dataset.h"
#include "oneflow/user/data/ofrecord_parser.h"
#include "oneflow/user/data/random_shuffle_dataset.h"
#include "oneflow/user/data/batch_dataset.h"

namespace oneflow {
namespace data {

class OFRecordDataReader final : public DataReader<TensorBuffer> {
 public:
  OFRecordDataReader(user_op::KernelInitContext* ctx) : DataReader<TensorBuffer>(ctx) {
    batch_size_ = ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape().elem_cnt();
    if (auto* pool = TensorBufferPool::TryGet()) { pool->IncreasePoolSizeByBase(batch_size_); }
    loader_.reset(new OFRecordDataset(ctx));
    if (ctx->Attr<bool>("random_shuffle")) {
      loader_.reset(new RandomShuffleDataset<TensorBuffer>(ctx, std::move(loader_)));
    }
    loader_.reset(new BatchDataset<TensorBuffer>(batch_size_, std::move(loader_)));
    parser_.reset(new OFRecordParser());
    StartLoadThread();
  }

  ~OFRecordDataReader() override {
    if (auto* pool = TensorBufferPool::TryGet()) { pool->DecreasePoolSizeByBase(batch_size_); }
  }

 protected:
  using DataReader<TensorBuffer>::loader_;
  using DataReader<TensorBuffer>::parser_;

 private:
  size_t batch_size_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_OFRECORD_DATA_READER_H_
