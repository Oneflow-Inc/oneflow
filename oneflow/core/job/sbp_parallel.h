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
#ifndef ONEFLOW_CORE_JOB_SBP_PARALLEL_H_
#define ONEFLOW_CORE_JOB_SBP_PARALLEL_H_

#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/job/sbp_infer_hint.h"

namespace oneflow {

inline bool operator!=(const cfg::SbpParallel& lhs, const cfg::SbpParallel& rhs) {
  return !(lhs == rhs);
}

inline bool operator!=(const cfg::SbpSignature& lhs, const cfg::SbpSignature& rhs) {
  return !(lhs == rhs);
}

inline bool operator!=(const cfg::ParallelDistribution& lhs, const cfg::ParallelDistribution& rhs) {
  return !(lhs == rhs);
}

cfg::SbpParallel GetDualSbpParallel(const cfg::SbpParallel&);

bool IsSbpSignatureContaining(const cfg::SbpSignature& bigger, const cfg::SbpSignature& smaller);

void FilterSbpSignatureList(const cfg::SbpSignatureList& sbp_sig_list,
                            const cfg::SbpSignature& sbp_sig_conf,
                            cfg::SbpSignatureList* filtered_sbp_sig_list);

void SortSbpSignatureListByCopyCost(
    const cfg::SbpSignatureList& sbp_sig_list, const PbRpf<std::string>& ibns,
    const std::function<Maybe<const SbpInferHint*>(const std::string&)>& SbpInferHint4Ibn,
    const std::function<int32_t(const cfg::SbpSignature&)>& OrderValue4SbpSig,
    std::vector<const cfg::SbpSignature*>* sorted_sbp_signatures);

bool IsValidSbpParallelString(const std::string& sbp_str);
bool ParseSbpParallelFromString(const std::string& sbp_str, cfg::SbpParallel* sbp_parallel);
std::string SbpParallelToString(const cfg::SbpParallel& sbp_parallel);
std::string ParallelDistributionToString(const Symbol<cfg::ParallelDistribution> nd_sbp);

void SbpSignatureToParallelDistributionSignature(
    const cfg::SbpSignature& sbp_signature, cfg::ParallelDistributionSignature* nd_sbp_signature);
template<typename ParallelDistributionSignatureT>
void ParallelDistributionSignatureToSbpSignature(
    const ParallelDistributionSignatureT& nd_sbp_signature, cfg::SbpSignature* sbp_signature);
void CheckSbpSignatureAndParallelDistributionEquals(
    const cfg::SbpSignature& sbp_sig, const cfg::ParallelDistributionSignature& nd_sbp_sig);
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::SbpSignature> {
  size_t operator()(const oneflow::SbpSignature& signature) const {
    std::string serialized_string;
    signature.SerializeToString(&serialized_string);
    return std::hash<std::string>()(serialized_string);
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_JOB_SBP_PARALLEL_H_
