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
#ifndef ONEFLOW_USER_DATA_COCO_DATASET_H_
#define ONEFLOW_USER_DATA_COCO_DATASET_H_

#include "oneflow/user/data/dataset.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {
namespace data {

struct COCOImage {
  TensorBuffer data;
  int64_t index;
  int64_t id;
  int32_t height;
  int32_t width;
};

class COCOMeta;

class COCODataset final : public RandomAccessDataset<COCOImage> {
 public:
  using LoadTargetShdPtr = std::shared_ptr<COCOImage>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  COCODataset(user_op::KernelInitContext* ctx, const std::shared_ptr<const COCOMeta>& meta)
      : meta_(meta), session_id_(ctx->Attr<int64_t>("session_id")) {}
  ~COCODataset() = default;

  LoadTargetShdPtrVec At(int64_t index) const override;
  size_t Size() const override;

 private:
  std::shared_ptr<const COCOMeta> meta_;
  int64_t session_id_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_COCO_DATASET_H_
