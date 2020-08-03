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
#include <iostream>

namespace oneflow {
namespace data {

class OFRecordDataReader final : public DataReader<TensorBuffer> {
 public:
  OFRecordDataReader(user_op::KernelInitContext* ctx) : DataReader<TensorBuffer>(ctx) {
    loader_.reset(new OFRecordDataset(ctx));
    parser_.reset(new OFRecordParser());
    if (ctx->Attr<bool>("random_shuffle")) {
      loader_.reset(new RandomShuffleDataset<TensorBuffer>(ctx, std::move(loader_)));
    }
    int32_t batch_size = ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape().elem_cnt();
    loader_.reset(new BatchDataset<TensorBuffer>(batch_size, std::move(loader_)));
    StartLoadThread();
  }
  ~OFRecordDataReader() = default;

 protected:
  using DataReader<TensorBuffer>::loader_;
  using DataReader<TensorBuffer>::parser_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_OFRECORD_DATA_READER_H_
