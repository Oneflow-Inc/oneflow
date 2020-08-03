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
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {
namespace data {

template<typename LoadTarget>
class BatchDataset final : public Dataset<LoadTarget> {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  BatchDataset(int32_t batch_size, std::unique_ptr<Dataset<LoadTarget>>&& data_set)
      : batch_size_(batch_size), loader_(std::move(data_set)) {}
  ~BatchDataset() = default;

  LoadTargetPtrList Next() override {
    LoadTargetPtrList ret;
    ret.reserve(batch_size_);
    for (int32_t i = 0; i < batch_size_; ++i) {
      LoadTargetPtrList tmp = loader_->Next();
      CHECK_EQ(tmp.size(), 1);
      ret.push_back(std::move(tmp.at(0)));
    }
    return ret;
  }

 private:
  int32_t batch_size_;
  std::unique_ptr<Dataset<LoadTarget>> loader_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_BATCH_DATASET_H_
