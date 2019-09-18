#ifndef ONEFLOW_CORE_JOB_SBP_PARALLEL_H_
#define ONEFLOW_CORE_JOB_SBP_PARALLEL_H_

#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/sbp_infer_hint.h"

namespace oneflow {

bool operator==(const SbpParallel& lhs, const SbpParallel& rhs);
bool operator!=(const SbpParallel& lhs, const SbpParallel& rhs);
bool operator==(const SbpSignature& lhs, const SbpSignature& rhs);
bool operator!=(const SbpSignature& lhs, const SbpSignature& rhs);

SbpParallel GetDualSbpParallel(const SbpParallel&);

bool IsSbpSignatureContaining(const SbpSignature& bigger, const SbpSignature& smaller);

void FilterSbpSignatureList(const SbpSignatureList& sbp_sig_list, const SbpSignature& sbp_sig_conf,
                            SbpSignatureList* filtered_sbp_sig_list);

void SortSbpSignatureListByCopyCost(
    const SbpSignatureList& sbp_sig_list, const PbRpf<std::string>& ibns,
    const std::function<Maybe<const SbpInferHint*>(const std::string&)>& SbpInferHint4Ibn,
    const std::function<int32_t(const SbpSignature&)>& OrderValue4SbpSig,
    std::vector<const SbpSignature*>* sorted_sbp_signatures);

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
