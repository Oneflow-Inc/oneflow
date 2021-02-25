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
#ifndef ONEFLOW_CORE_KERNEL_RUNTIME_BLOB_SHAPE_INFER_HELPER_H_
#define ONEFLOW_CORE_KERNEL_RUNTIME_BLOB_SHAPE_INFER_HELPER_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/op_infer_cache.h"

namespace oneflow {

class Blob;
class RtBlobDesc;

class RuntimeBlobShapeInferHelper final {
 public:
  RuntimeBlobShapeInferHelper(const OperatorConf& op_conf, const KernelConf& kernel_conf,
                              const JobDesc* job_desc);
  ~RuntimeBlobShapeInferHelper() = default;

  void InferShape(std::function<Blob*(const std::string&)> BnInOp2Blob);

 private:
  void UpdateInputBlobDescs7OpInferCacheKey(std::function<Blob*(const std::string&)> BnInOp2Blob);
  BlobDesc* BlobDesc4BnInOp(const std::string& bn_in_op, const RtBlobDesc& rt_blob_desc);

  std::shared_ptr<Operator> op_;
  HashSet<std::string> ibns_;
  HashMap<std::string, std::unique_ptr<BlobDesc>> bn_in_op2blob_desc_;
  HashMap<std::string, std::unique_ptr<BlobDesc>> bn_in_op2logical_blob_desc_;
  ParallelContext parallel_ctx_;
  SbpSignature sbp_signature_;
  OpInferCacheKey op_infer_cache_key_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RUNTIME_BLOB_SHAPE_INFER_HELPER_H_
