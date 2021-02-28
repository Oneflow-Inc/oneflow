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
#include "oneflow/core/kernel/runtime_blob_shape_infer_helper.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/cached_caller.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

RuntimeBlobShapeInferHelper::RuntimeBlobShapeInferHelper(const OperatorConf& op_conf,
                                                         const KernelConf& kernel_conf,
                                                         const JobDesc* job_desc) {
  op_ = ConstructOp(op_conf, job_desc);
  auto* map = sbp_signature_.mutable_bn_in_op2sbp_parallel();
  op_->ForEachBnInOp([&](const std::string& bn_in_op) {
    bn_in_op2blob_desc_[bn_in_op].reset();
    (*map)[bn_in_op].mutable_split_parallel()->set_axis(0);
  });
  parallel_ctx_.set_parallel_id(0);
  parallel_ctx_.set_parallel_num(1);
  op_infer_cache_key_.job_desc = job_desc;
  op_infer_cache_key_.op_conf_sym = op_->GetOpConfWithoutOpNameAndLbn();
  op_infer_cache_key_.ibn_idx2shape_sym.resize(op_->input_bns().size());
  op_infer_cache_key_.dtype_signature_sym = SymbolOf(kernel_conf.dtype_signature());
}

void RuntimeBlobShapeInferHelper::UpdateInputBlobDescs7OpInferCacheKey(
    std::function<Blob*(const std::string&)> BnInOp2Blob) {
  auto ResetBlobDescAndGetShapeSym = [&](const std::string& ibn) -> Symbol<Shape> {
    const Blob* blob = BnInOp2Blob(ibn);
    if (blob == nullptr) { return Symbol<Shape>(); }
    BlobDesc* blob_desc = BlobDesc4BnInOp(ibn, blob->blob_desc());
    blob_desc->mut_shape().LeftOnesExtendedAssign(blob->shape());
    return SymbolOf(blob_desc->shape());
  };
  const auto& input_bns = op_->input_bns();
  FOR_RANGE(int, i, 0, input_bns.size()) {
    op_infer_cache_key_.ibn_idx2shape_sym.at(i) = ResetBlobDescAndGetShapeSym(input_bns.Get(i));
  }
}

BlobDesc* RuntimeBlobShapeInferHelper::BlobDesc4BnInOp(const std::string& bn_in_op,
                                                       const RtBlobDesc& rt_blob_desc) {
  BlobDesc* blob_desc = bn_in_op2blob_desc_.at(bn_in_op).get();
  if (blob_desc != nullptr) { return blob_desc; }
  blob_desc =
      new BlobDesc(rt_blob_desc.body(), rt_blob_desc.is_tensor_list(), rt_blob_desc.is_dynamic());
  bn_in_op2blob_desc_.at(bn_in_op).reset(blob_desc);
  return blob_desc;
}

void RuntimeBlobShapeInferHelper::InferShape(std::function<Blob*(const std::string&)> BnInOp2Blob) {
  UpdateInputBlobDescs7OpInferCacheKey(BnInOp2Blob);
  auto Infer = [&](const OpInferCacheKey& key) -> std::shared_ptr<const OpInferCacheValue> {
    auto CachedBlobDesc4BnInOp = WithResultCached([&](const std::string& bn_in_op) -> BlobDesc* {
      const Blob* blob = BnInOp2Blob(bn_in_op);
      if (blob == nullptr) { return nullptr; }
      return BlobDesc4BnInOp(bn_in_op, blob->blob_desc());
    });
    CHECK_JUST(op_->InferOutBlobDescsIf(CachedBlobDesc4BnInOp, &parallel_ctx_, &sbp_signature_));
    auto* ret = new OpInferCacheValue();
    ret->obn_idx2shape_sym.resize(op_->output_bns().size());
    FOR_RANGE(int, i, 0, op_->output_bns().size()) {
      const auto& obn = op_->output_bns().Get(i);
      const auto& blob_desc = bn_in_op2blob_desc_.at(obn);
      ret->obn_idx2shape_sym.at(i).reset(blob_desc->shape());
      auto* blob = BnInOp2Blob(obn);
      if (blob == nullptr) { continue; }
      CHECK_EQ(blob->data_type(), blob_desc->data_type());
      CHECK_EQ(blob->blob_desc().is_dynamic(), blob_desc->is_dynamic());
    }
    return std::shared_ptr<const OpInferCacheValue>(ret);
  };
  size_t cache_size = Global<ResourceDesc, ForSession>::Get()->thread_local_cache_max_size();
  const auto& shape_infer_ret = ThreadLocalCachedCall(cache_size, Infer, op_infer_cache_key_);
  const auto& obn_idx2shape_sym = shape_infer_ret->obn_idx2shape_sym;
  FOR_RANGE(int, i, 0, op_->output_bns().size()) {
    const auto& obn = op_->output_bns().Get(i);
    auto* blob = BnInOp2Blob(obn);
    if (blob == nullptr) { continue; }
    if (blob->blob_desc().is_dynamic()) {
      blob->mut_shape_view()->set_shape(*obn_idx2shape_sym.at(i));
    } else {
      CHECK(*obn_idx2shape_sym.at(i) == blob->static_shape());
    }
  }
}

}  // namespace oneflow
