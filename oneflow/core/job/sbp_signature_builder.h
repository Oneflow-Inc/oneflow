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
#ifndef ONEFLOW_CORE_JOB_SBP_SIGNATURE_BUILDER_H_
#define ONEFLOW_CORE_JOB_SBP_SIGNATURE_BUILDER_H_

#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

class SplitSbpSignatureListBuilder final {
 public:
  SplitSbpSignatureListBuilder(const SplitSbpSignatureListBuilder&) = default;
  explicit SplitSbpSignatureListBuilder(const SbpSignature& sbp_signature_template)
      : sbp_signature_template_(sbp_signature_template), num_axes_(0) {
    CheckTemplate();
  }
  ~SplitSbpSignatureListBuilder() = default;

  SplitSbpSignatureListBuilder&& SetNumAxes(int64_t num_axes);
  void Build(SbpSignatureList* list) const;

 private:
  void CheckTemplate();

  SbpSignature sbp_signature_template_;
  int64_t num_axes_;
};

class SbpSignatureBuilder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SbpSignatureBuilder);
  SbpSignatureBuilder() = default;
  ~SbpSignatureBuilder() = default;

  // split
  SbpSignatureBuilder&& Split(const std::string& bn_in_op, int64_t axis);
  SbpSignatureBuilder&& Split(const PbRpf<std::string>& bns, int64_t axis);
  SbpSignatureBuilder&& Split(const std::initializer_list<std::string>& bns, int64_t axis);

  // broadcast
  SbpSignatureBuilder&& Broadcast(const std::string& bn_in_op);
  SbpSignatureBuilder&& Broadcast(const PbRpf<std::string>& bns);
  SbpSignatureBuilder&& Broadcast(const std::initializer_list<std::string>& bns);

  // partial_sum
  SbpSignatureBuilder&& PartialSum(const std::string& bn_in_op);
  SbpSignatureBuilder&& PartialSum(const PbRpf<std::string>& bns);
  SbpSignatureBuilder&& PartialSum(const std::initializer_list<std::string>& bns);

  SplitSbpSignatureListBuilder MakeSplitSignatureListBuilder(int64_t num_axes) const;
  void Build(SbpSignature* ret) const { *ret = sbp_signature_; }

 private:
  SbpSignature sbp_signature_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SBP_SIGNATURE_BUILDER_H_
