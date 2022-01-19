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

#include "oneflow/user/data/data_reader.h"
#include "oneflow/user/data/onerec_dataset.h"
#include "oneflow/user/data/onerec_parser.h"
#include "oneflow/user/data/random_shuffle_dataset.h"
#include "oneflow/user/data/batch_random_shuffle_dataset.h"
#include "oneflow/user/data/batch_dataset.h"
#include <iostream>

namespace oneflow {
namespace data {

class OneRecDataReader final : public DataReader<TensorBuffer> {
 public:
  OneRecDataReader(user_op::KernelInitContext* ctx) : DataReader<TensorBuffer>(ctx) {
    const int32_t batch_size = ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape().elem_cnt();
    const auto random_shuffle = ctx->Attr<bool>("random_shuffle");
    parser_.reset(new OneRecParser());
    if (random_shuffle) {
      const auto mode = ctx->Attr<std::string>("shuffle_mode");
      if (mode == "batch") {
        loader_.reset(new OneRecDataset(ctx, batch_size));
        loader_.reset(new BatchRandomShuffleDataset<TensorBuffer>(ctx, std::move(loader_)));
      } else if (mode == "instance") {
        loader_.reset(new OneRecDataset(ctx, 1));
        loader_.reset(new RandomShuffleDataset<TensorBuffer>(ctx, std::move(loader_)));
        loader_.reset(new BatchDataset<TensorBuffer>(batch_size, std::move(loader_)));
      } else {
        UNIMPLEMENTED();
      }
    } else {
      loader_.reset(new OneRecDataset(ctx, batch_size));
    }
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
