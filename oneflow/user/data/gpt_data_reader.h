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
#ifndef ONEFLOW_USER_DATA_GPT_DATA_READER_H_
#define ONEFLOW_USER_DATA_GPT_DATA_READER_H_

#include "oneflow/user/data/data_reader.h"
#include "oneflow/core/common/tensor_buffer.h"

namespace oneflow {

namespace data {

class GPTDataReader final : public DataReader<TensorBuffer> {
 public:
  GPTDataReader(user_op::KernelInitContext* ctx);
  ~GPTDataReader() = default;

 protected:
  using DataReader<TensorBuffer>::loader_;
  using DataReader<TensorBuffer>::parser_;

 private:
  std::vector<size_t> GetSplitDocIndices(const std::vector<int64_t>& split_sizes,
                                         int64_t split_index, size_t num_docs) const;
  size_t GetDistributedBatchSize(size_t batch_size, const Shape& hierarchy,
                                 const ParallelDistribution& parallel_dist) const;
};

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_GPT_DATA_READER_H_
