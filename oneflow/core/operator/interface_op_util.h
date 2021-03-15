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
#ifndef ONEFLOW_CORE_OPERATOR_INTERFACE_OP_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_INTERFACE_OP_UTIL_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

struct InterfaceOpUtil final {
  static Maybe<void> InferOutBlobDesc(const InterfaceBlobConf& blob_conf, BlobDesc* out_blob_desc,
                                      const ParallelContext* parallel_ctx);
  static Maybe<void> InferLogicalOutBlobDesc(const InterfaceBlobConf& blob_conf,
                                             BlobDesc* out_blob_desc,
                                             const ParallelDesc& parallel_desc);
  static Maybe<void> GetInputLikeOpSbpSignature(const InterfaceBlobConf& blob_conf,
                                                const PbRpf<std::string>& input_bns,
                                                const PbRpf<std::string>& output_bns,
                                                SbpSignature* sbp_signature);
  static Maybe<void> GetOutputLikeOpSbpSignature(const InterfaceBlobConf& blob_conf,
                                                 const PbRpf<std::string>& input_bns,
                                                 const PbRpf<std::string>& output_bns,
                                                 SbpSignature* sbp_signature);
  static Maybe<void> InitBlobConf(InterfaceBlobConf* blob_conf,
                                  const ParallelBlobConf& parallel_blob_conf);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_INTERFACE_OP_UTIL_H_
