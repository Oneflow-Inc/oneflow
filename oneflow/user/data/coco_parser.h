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
#ifndef ONEFLOW_USER_DATA_COCO_PARSER_H_
#define ONEFLOW_USER_DATA_COCO_PARSER_H_

#include "oneflow/user/data/parser.h"
#include "oneflow/user/data/coco_dataset.h"

namespace oneflow {
namespace data {

class COCOMeta;

class COCOParser final : public Parser<COCOImage> {
 public:
  using Base = Parser<COCOImage>;
  using SampleType = typename Base::SampleType;
  using BatchType = typename Base::BatchType;

  COCOParser(const std::shared_ptr<const COCOMeta>& meta) : meta_(meta){};
  ~COCOParser() = default;

  void Parse(BatchType& batch_data, user_op::KernelComputeContext* ctx) override;

 private:
  std::shared_ptr<const COCOMeta> meta_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_COCO_PARSER_H_
