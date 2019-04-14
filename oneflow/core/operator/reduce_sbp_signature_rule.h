#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_SBP_SIGNATURE_RULE_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_SBP_SIGNATURE_RULE_H_

#include "oneflow/core/job/sbp_signature_rule.h"

namespace oneflow {

void GetReduceSbpSignatureRules(const Operator* op, const std::string& data_ibn,
                                const HashSet<int64_t>& reduced_axes,
                                std::vector<std::unique_ptr<const SbpSignatureRule>>* rules);

void GetReduceGradSbpSignatureRules(const Operator* op, const std::string& like_ibn,
                                    const HashSet<int64_t>& reduced_axes,
                                    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules);
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_SBP_SIGNATURE_RULE_H_
