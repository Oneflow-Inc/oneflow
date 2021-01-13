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
#include "oneflow/core/framework/op_arg_util.h"

namespace oneflow {

namespace compatible_py {

Maybe<OpArgBlobAttribute> GetOpArgBlobAttribute(const OpAttribute& op_attribute,
                                                const std::string& bn_in_op) {
  if (!op_attribute.has_batch_axis_signature()) { return std::shared_ptr<OpArgBlobAttribute>(); }
  if (!op_attribute.has_logical_blob_desc_signature()) {
    return std::shared_ptr<OpArgBlobAttribute>();
  }
  auto& batch_axis_signature_map = op_attribute.batch_axis_signature().bn_in_op2batch_axis();
  auto& blob_desc_signature_map = op_attribute.logical_blob_desc_signature().bn_in_op2blob_desc();
  auto& arg_signature_map = op_attribute.arg_signature().bn_in_op2lbi();
  auto& lbi = arg_signature_map.at(bn_in_op);
  std::shared_ptr<cfg::OptInt64> batch_axis = std::make_shared<cfg::OptInt64>();
  if (batch_axis_signature_map.find(bn_in_op) != batch_axis_signature_map.end()) {
    batch_axis.reset(new cfg::OptInt64(batch_axis_signature_map.at(bn_in_op)));
  }
  std::shared_ptr<cfg::BlobDescProto> blob_desc = std::make_shared<cfg::BlobDescProto>();
  if (blob_desc_signature_map.find(bn_in_op) != blob_desc_signature_map.end()) {
    blob_desc.reset(new cfg::BlobDescProto(blob_desc_signature_map.at(bn_in_op)));
  }
  return std::make_shared<OpArgBlobAttribute>(batch_axis, blob_desc,
                                              lbi.op_name() + "/" + lbi.blob_name());
}

Maybe<OpArgParallelAttribute> GetOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol, const OpAttribute& op_attribute,
    const std::string& bn_in_op) {
  auto& sbp_signature_map = op_attribute.sbp_signature().bn_in_op2sbp_parallel();
  auto& mirrored_signature_map = op_attribute.mirrored_signature().bn_in_op2opt_mirrored_parallel();
  std::shared_ptr<cfg::SbpParallel> sbp_parallel = std::make_shared<cfg::SbpParallel>();
  if (sbp_signature_map.find(bn_in_op) != sbp_signature_map.end()) {
    sbp_parallel.reset(new cfg::SbpParallel(sbp_signature_map.at(bn_in_op)));
  }
  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel =
      std::make_shared<cfg::OptMirroredParallel>();
  if (mirrored_signature_map.find(bn_in_op) != mirrored_signature_map.end()) {
    opt_mirrored_parallel.reset(new cfg::OptMirroredParallel(mirrored_signature_map.at(bn_in_op)));
  }
  return std::make_shared<OpArgParallelAttribute>(parallel_desc_symbol, sbp_parallel,
                                                  opt_mirrored_parallel);
}

Maybe<OpArgParallelAttribute> MakeMirroredOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  std::shared_ptr<cfg::SbpParallel> sbp_parallel = std::make_shared<cfg::SbpParallel>();
  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel =
      std::make_shared<cfg::OptMirroredParallel>();
  opt_mirrored_parallel->mutable_mirrored_parallel();
  return std::make_shared<OpArgParallelAttribute>(parallel_desc_symbol, sbp_parallel,
                                                  opt_mirrored_parallel);
}

Maybe<OpArgParallelAttribute> MakeBroadcastOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  std::shared_ptr<cfg::SbpParallel> sbp_parallel = std::make_shared<cfg::SbpParallel>();
  sbp_parallel->mutable_broadcast_parallel();
  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel =
      std::make_shared<cfg::OptMirroredParallel>();
  return std::make_shared<OpArgParallelAttribute>(parallel_desc_symbol, sbp_parallel,
                                                  opt_mirrored_parallel);
}

}  // namespace compatible_py

}  // namespace oneflow
