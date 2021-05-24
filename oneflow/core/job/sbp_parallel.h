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
#include "oneflow/core/job/sbp_infer_hint.h"

namespace oneflow {

bool operator==(const SbpParallel& lhs, const SbpParallel& rhs);
bool operator!=(const SbpParallel& lhs, const SbpParallel& rhs);
bool operator==(const SbpSignature& lhs, const SbpSignature& rhs);
bool operator!=(const SbpSignature& lhs, const SbpSignature& rhs);

bool operator==(const ParallelDistribution& lhs, const ParallelDistribution& rhs);
bool operator!=(const ParallelDistribution& lhs, const ParallelDistribution& rhs);
bool operator==(const ParallelDistributionSignature& lhs, const ParallelDistributionSignature& rhs);
bool operator!=(const ParallelDistributionSignature& lhs, const ParallelDistributionSignature& rhs);

SbpParallel GetDualSbpParallel(const SbpParallel&);

bool IsSbpSignatureContaining(const SbpSignature& bigger, const SbpSignature& smaller);

void FilterSbpSignatureList(const SbpSignatureList& sbp_sig_list, const SbpSignature& sbp_sig_conf,
                            SbpSignatureList* filtered_sbp_sig_list);

void SortSbpSignatureListByCopyCost(
    const SbpSignatureList& sbp_sig_list, const PbRpf<std::string>& ibns,
    const std::function<Maybe<const SbpInferHint*>(const std::string&)>& SbpInferHint4Ibn,
    const std::function<int32_t(const SbpSignature&)>& OrderValue4SbpSig,
    std::vector<const SbpSignature*>* sorted_sbp_signatures);

bool IsValidSbpParallelString(const std::string& sbp_str);
bool ParseSbpParallelFromString(const std::string& sbp_str, SbpParallel* sbp_parallel);
std::string SbpParallelToString(const SbpParallel& sbp_parallel);

void SbpSignatureToParallelDistributionSignature(
    const SbpSignature& sbp_signature,
    ParallelDistributionSignature* parallel_distribution_signature);
void ParallelDistributionSignatureToSbpSignature(
    const ParallelDistributionSignature& parallel_distribution_signature,
    SbpSignature* sbp_signature);
void CheckSbpSignatureAndParallelDistributionEquals(
    const SbpSignature& sbp_sig, const ParallelDistributionSignature& parallel_distribution_sig);

template<typename SbpParallelT>
struct HashSbpParallel {
  size_t operator()(const SbpParallelT& sbp_parallel) const {
    size_t hash_value = static_cast<size_t>(sbp_parallel.parallel_type_case());
    if (sbp_parallel.has_split_parallel()) {
      hash_value ^= sbp_parallel.split_parallel().axis();
    } else if (sbp_parallel.has_broadcast_parallel()) {
      // Do nothing.
    } else if (sbp_parallel.has_partial_sum_parallel()) {
      // Do nothing.
    } else {
      UNIMPLEMENTED();
    }
    return hash_value;
  }
};

template<typename ParallelDistributionT, typename SbpParallelT>
struct HashParallelDistribution {
  size_t operator()(const ParallelDistributionT& parallel_distribution) const {
    const auto& sbp_hash_functor = std::hash<SbpParallelT>();
    size_t hash_value = 0;
    for (const auto& sbp_parallel : parallel_distribution.sbp_parallel()) {
      hash_value ^= sbp_hash_functor(sbp_parallel);
    }
    return hash_value;
  }
};

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

template<>
struct hash<oneflow::SbpParallel>
  : public oneflow::HashSbpParallel<oneflow::SbpParallel> { };

template<>
struct hash<oneflow::cfg::SbpParallel>
  : public oneflow::HashSbpParallel<oneflow::cfg::SbpParallel> { };

template<>
struct hash<oneflow::ParallelDistribution>
  : public oneflow::HashParallelDistribution<oneflow::ParallelDistribution, oneflow::SbpParallel> { };

template<>
struct hash<oneflow::cfg::ParallelDistribution>
  : public oneflow::HashParallelDistribution<oneflow::cfg::ParallelDistribution, oneflow::cfg::SbpParallel> { };

}  // namespace std

#endif  // ONEFLOW_CORE_JOB_SBP_PARALLEL_H_
