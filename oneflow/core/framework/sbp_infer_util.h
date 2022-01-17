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

#ifndef ONEFLOW_CORE_FRAMEWORK_SBP_INFER_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_SBP_INFER_UTIL_H_

#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

double GetValidMaxCopyCost();

Maybe<std::string> NdSbpSignatureListAsString(
    const std::vector<cfg::NdSbpSignature>& nd_sbp_sig_list, const PbRpf<std::string>& inputs,
    const PbRpf<std::string>& outputs);

void ResizeNdSbpSignature(cfg::NdSbpSignature& nd_sbp_sig, int32_t size);

void SetNdSbpSignature(cfg::NdSbpSignature* nd_sbp_signature,
                       const cfg::SbpSignature& sbp_signature, int32_t sbp_axis);

void DfsGetNdSbpSignature(cfg::NdSbpSignature& nd_sbp_sig, int32_t depth, int32_t dims,
                          const cfg::SbpSignatureList& sbp_sig_list,
                          std::vector<cfg::NdSbpSignature>* nd_sbp_sig_list);

// TODO: unified lazy and eager boxing
// Compute eager copy cost
Maybe<double> ComputeEagerCopyCostBetweenNdSbp(const cfg::NdSbp& producer_sbp_parallel,
                                               const cfg::NdSbp& consumer_sbp_parallel,
                                               const BlobDesc& logical_blob_desc,
                                               const ParallelDesc& producer_parallel_desc,
                                               const ParallelDesc& consumer_parallel_desc,
                                               bool is_same_sbp);

// Compute lazy copy cost
Maybe<double> ComputeLazyCopyCostBetweenNdSbp(const cfg::NdSbp& producer_sbp_parallel,
                                              const cfg::NdSbp& consumer_sbp_parallel,
                                              const BlobDesc& logical_blob_desc,
                                              const ParallelDesc& producer_parallel_desc,
                                              const ParallelDesc& consumer_parallel_desc,
                                              bool is_same_sbp);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SBP_INFER_UTIL_H_
