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
#ifndef ONEFLOW_USER_DATA_BATCH_DATASET_H_
#define ONEFLOW_USER_DATA_BATCH_DATASET_H_

#include "oneflow/user/data/dataset.h"

namespace oneflow {
namespace data {

template<typename LoadTarget>
class BatchDataset final : public Dataset<LoadTarget> {
 public:
  using Base = Dataset<LoadTarget>;
  using SampleType = typename Base::SampleType;
  using BatchType = typename Base::BatchType;

  BatchDataset(int32_t batch_size, std::unique_ptr<Dataset<LoadTarget>>&& dataset)
      : batch_size_(batch_size), nested_ds_(std::move(dataset)) {}
  ~BatchDataset() = default;

  BatchType Next() override {
    BatchType batch;
    batch.reserve(batch_size_);
    for (size_t i = 0; i < batch_size_; ++i) {
      BatchType tmp = nested_ds_->Next();
      CHECK_EQ(tmp.size(), 1);
      batch.push_back(std::move(tmp[0]));
    }
    return batch;
  }

 private:
  int32_t batch_size_;
  std::unique_ptr<Dataset<LoadTarget>> nested_ds_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_BATCH_DATASET_H_
