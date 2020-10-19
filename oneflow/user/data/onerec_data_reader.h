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
#ifndef ONEFLOW_CUSTOMIZED_DATA_ONEREC_DATA_READER_H_
#define ONEFLOW_CUSTOMIZED_DATA_ONEREC_DATA_READER_H_

#include "oneflow/customized/data/data_reader.h"
#include "oneflow/customized/data/onerec_dataset.h"
#include "oneflow/customized/data/onerec_parser.h"
#include "oneflow/customized/data/random_shuffle_dataset.h"
#include "oneflow/customized/data/batch_dataset.h"
#include <iostream>

namespace oneflow {
namespace data {

class OneRecDataReader final : public DataReader<TensorBuffer> {
 public:
  OneRecDataReader(user_op::KernelInitContext* ctx) : DataReader<TensorBuffer>(ctx) {
    loader_.reset(new OneRecDataset(ctx));
    parser_.reset(new OneRecParser());
    if (ctx->Attr<bool>("random_shuffle")) {
      loader_.reset(new RandomShuffleDataset<TensorBuffer>(ctx, std::move(loader_)));
    }
    int32_t batch_size = ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape().elem_cnt();
    loader_.reset(new BatchDataset<TensorBuffer>(batch_size, std::move(loader_)));
    StartLoadThread();
  }
  ~OneRecDataReader() = default;

 protected:
  using DataReader<TensorBuffer>::loader_;
  using DataReader<TensorBuffer>::parser_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_ONEREC_DATA_READER_H_
