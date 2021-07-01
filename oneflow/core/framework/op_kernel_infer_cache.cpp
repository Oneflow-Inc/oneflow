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
#include "oneflow/core/framework/op_kernel_infer_cache.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace user_op {

OpKernelInferCache::OpKernelInferCache(const KernelConf& kernel_conf, const JobDesc& job_desc) {
  const OperatorConf& op_conf = kernel_conf.op_attribute().op_conf();
  std::shared_ptr<Operator> op = ConstructOp(op_conf);
  cache_key_.job_desc = &job_desc;
  cache_key_.op_conf_sym = op->GetOpConfWithoutOpNameAndLbn();
  cache_key_.ibn_idx2shape_sym.resize(op->input_bns().size());
  cache_key_.dtype_signature_sym = SymbolOf(kernel_conf.dtype_signature());
}

bool OpKernelInferCache::IsCacheHit() const {
  size_t hash_value = std::hash<KeyType>()(cache_key_);
  HashEqTraitPtr<const KeyType> ptr_wrapper(&cache_key_, hash_value);
  return cached_key2value_.find(ptr_wrapper) != cached_key2value_.end();
}

OpKernelInferCache::ValueType OpKernelInferCache::GetCacheValue() const {
  size_t hash_value = std::hash<KeyType>()(cache_key_);
  HashEqTraitPtr<const KeyType> ptr_wrapper(&cache_key_, hash_value);
  CHECK(cached_key2value_.find(ptr_wrapper) != cached_key2value_.end());
  return cached_key2value_.at(ptr_wrapper);
}

void OpKernelInferCache::UpdateCacheKey(KernelInferContext* ctx) {
  auto GetSymbolOfShape = [&](const std::string& arg_name, int32_t arg_index) -> Symbol<Shape> {
    Shape shape;
    ctx->ShapeView4ArgNameAndIndex(arg_name, arg_index).ToShape(&shape);
    return SymbolOf(shape);
  };
  const auto& inputs = ctx->inputs();
  FOR_RANGE(int, i, 0, inputs.size()) {
    const auto& arg_pair = inputs.at(i);
    cache_key_.ibn_idx2shape_sym.at(i) = GetSymbolOfShape(arg_pair.first, arg_pair.second);
  }
}

void OpKernelInferCache::UpdateCacheValue(KernelInferContext* ctx) {
  if (cached_key2value_.size() >= max_size_) { Reset(); }
  auto* cache_value = new OpInferCacheValue();
  cache_value->obn_idx2shape_sym.resize(ctx->outputs().size());
  FOR_RANGE(int, i, 0, ctx->outputs().size()) {
    const auto& out_arg_pair = ctx->outputs().at(i);
    const ShapeView& out_shape_view =
        ctx->ShapeView4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
    Shape out_shape;
    out_shape_view.ToShape(&out_shape);
    cache_value->obn_idx2shape_sym.at(i).reset(out_shape);
  }
  KeyType* new_key = new KeyType(cache_key_);
  key_storage_.emplace_back(new_key);
  size_t hash_value = std::hash<KeyType>()(cache_key_);
  HashEqTraitPtr<const KeyType> ptr_wrapper(new_key, hash_value);
  CHECK(cached_key2value_.emplace(ptr_wrapper, ValueType(cache_value)).second);
}

void OpKernelInferCache::Reset() {
  CHECK_EQ(cached_key2value_.size(), key_storage_.size());
  HashMap to_release_key2values;
  KeyStorage to_release_key_storage;
  std::swap(cached_key2value_, to_release_key2values);
  std::swap(key_storage_, to_release_key_storage);
  if (to_release_key2values.size() <= kReleaseInIndependentThreadThreshold) {
    to_release_key2values.clear();
    to_release_key_storage.clear();
  } else {
    std::thread(
        [](HashMap&& cache, KeyStorage&& key_storage) {
          cache.clear();
          key_storage.clear();
        },
        std::move(to_release_key2values), std::move(to_release_key_storage));
  }
}

}  // namespace user_op

}  // namespace oneflow
