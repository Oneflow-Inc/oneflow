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
#ifndef ONEFLOW_USER_DATA_BATCH_RANDOM_SHUFFLE_DATASET_H_
#define ONEFLOW_USER_DATA_BATCH_RANDOM_SHUFFLE_DATASET_H_

#include "oneflow/user/data/dataset.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {
namespace data {

template<typename LoadTarget>
class BatchRandomShuffleDataset final : public Dataset<LoadTarget> {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  BatchRandomShuffleDataset(user_op::KernelInitContext* ctx,
                            std::unique_ptr<Dataset<LoadTarget>>&& data_set)
      : loader_(std::move(data_set)) {
    // random
    seed_ = ctx->Attr<int64_t>("seed");
    if (seed_ == -1) { seed_ = NewRandomSeed(); }
    std::seed_seq seq({seed_});
    rand_engine_ = std::default_random_engine(seq);

    // fill buffer
    initial_buffer_fill_ = ctx->Attr<int32_t>("shuffle_buffer_size");
    for (int32_t i = 0; i < initial_buffer_fill_; ++i) {
      LoadTargetPtrList batch = loader_->Next();
      batch_buffer_.push_back(std::move(batch));
    }
  }
  ~BatchRandomShuffleDataset() = default;

  LoadTargetPtrList Next() override {
    LoadTargetPtrList ret = loader_->Next();
    std::uniform_int_distribution<> dis(0, batch_buffer_.size() - 1);
    const int offset = dis(rand_engine_);
    std::swap(batch_buffer_.at(offset), ret);
    return ret;
  }

 private:
  std::unique_ptr<Dataset<LoadTarget>> loader_;
  std::vector<LoadTargetPtrList> batch_buffer_;

  int32_t initial_buffer_fill_;

  std::default_random_engine rand_engine_;
  int64_t seed_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_BATCH_RANDOM_SHUFFLE_DATASET_H_
