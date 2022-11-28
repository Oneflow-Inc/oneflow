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
#include "oneflow/core/framework/framework.h"
namespace oneflow {

namespace user_op {

namespace {

class BCEWithLogitsReduceMeanKernelCache final : public user_op::OpKernelCache {
 public:
  BCEWithLogitsReduceMeanKernelCache(int64_t reduce_elem_cnt) : reduce_elem_cnt_(reduce_elem_cnt) {}
  ~BCEWithLogitsReduceMeanKernelCache() override = default;

  int64_t reduce_elem_cnt() const { return reduce_elem_cnt_; }

 private:
  const int64_t reduce_elem_cnt_;
};

std::shared_ptr<user_op::OpKernelCache> CreateBCEWithLogitsReduceMeanKernelCache(
    user_op::KernelCacheContext* ctx) {
  if (ctx->parallel_ctx().parallel_num() == 1) { return nullptr; }
  const int64_t reduce_elem_cnt =
      ctx->LogicalTensorDesc4ArgNameAndIndex("input", 0)->shape().elem_cnt();
  return std::make_shared<BCEWithLogitsReduceMeanKernelCache>(reduce_elem_cnt);
}

}  // namespace

}  // namespace user_op

}  // namespace oneflow
