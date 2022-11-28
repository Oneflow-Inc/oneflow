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
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

bool operator==(const SbpParallel& lhs, const SbpParallel& rhs);
inline bool operator!=(const SbpParallel& lhs, const SbpParallel& rhs) { return !(lhs == rhs); }

bool operator==(const NdSbp& lhs, const NdSbp& rhs);
inline bool operator!=(const NdSbp& lhs, const NdSbp& rhs) { return !(lhs == rhs); }

bool operator==(const SbpSignature& lhs, const SbpSignature& rhs);
inline bool operator!=(const SbpSignature& lhs, const SbpSignature& rhs) { return !(lhs == rhs); }

bool operator==(const NdSbpSignature& lhs, const NdSbpSignature& rhs);
inline bool operator!=(const NdSbpSignature& lhs, const NdSbpSignature& rhs) {
  return !(lhs == rhs);
}

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::SbpSignature> : public oneflow::SerializedHashPb<oneflow::SbpSignature> {};

template<>
struct hash<oneflow::NdSbpSignature> : public oneflow::SerializedHashPb<oneflow::NdSbpSignature> {};

}  // namespace std

namespace oneflow {

Maybe<Symbol<SbpParallel>> MakeSplitSbpParallel(int axis);
Maybe<Symbol<SbpParallel>> MakeBroadcastSbpParallel();
Maybe<Symbol<SbpParallel>> MakePartialSumSbpParallel();

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

bool ParseNdSbpFromStringList(const std::vector<std::string>& sbp_str_list, NdSbp* nd_sbp);
std::vector<std::string> NdSbpToStringList(const NdSbp& nd_sbp);

void SbpSignatureToNdSbpSignature(const SbpSignature& sbp_signature,
                                  NdSbpSignature* nd_sbp_signature);

void NdSbpSignatureToSbpSignature(const NdSbpSignature& nd_sbp_signature,
                                  SbpSignature* sbp_signature);

void CheckSbpSignatureAndNdSbpEquals(const SbpSignature& sbp_sig, const NdSbpSignature& nd_sbp_sig);

bool NdSbpAllSameSplitParallel(const NdSbp& nd_sbp);

// Print functions

Maybe<std::string> NdSbpSignatureToString(const NdSbpSignature& nd_sbp_signature,
                                          const std::vector<std::string>& inputs,
                                          const std::vector<std::string>& outputs);

Maybe<std::string> NdSbpSignatureToString(const NdSbpSignature& nd_sbp_signature,
                                          const PbRpf<std::string>& inputs,
                                          const PbRpf<std::string>& outputs);

Maybe<std::string> NdSbpSignatureListToString(const std::vector<NdSbpSignature>& nd_sbp_sig_list,
                                              const std::vector<std::string>& inputs,
                                              const std::vector<std::string>& outputs);

Maybe<std::string> NdSbpSignatureListToString(const std::vector<NdSbpSignature>& nd_sbp_sig_list,
                                              const PbRpf<std::string>& inputs,
                                              const PbRpf<std::string>& outputs);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SBP_PARALLEL_H_
