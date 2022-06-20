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

enum SbpInferRuleTag : int {
  kAllMatch = 1,   // All match first, then lowest cost
  kMatchAMAP = 2,  // Match as much as possible
  kMinCost = 3     // Lowest cost
};

enum Penalty4PartialInConsumerTag : int {
  kSlight = 1,  // Slight penalty
  kMiddle = 2,  // Make sure we do not select P in the consumer
  kStrict = 3   // Not allow a transfer to P
};

void NdSbpDimReduce(const ParallelDesc& parallel_desc, const NdSbp& nd_sbp,
                    ParallelDesc* reduced_parallel_desc, NdSbp* reduced_nd_sbp);

void InOutParallelDimReduce(const ParallelDesc& in_parallel_desc,
                            const ParallelDesc& out_parallel_desc, const NdSbp& in_nd_sbp,
                            const NdSbp& out_nd_sbp, ParallelDesc* reduced_in_parallel_desc,
                            ParallelDesc* reduced_out_parallel_desc, NdSbp* reduced_in_nd_sbp,
                            NdSbp* reduced_out_nd_sbp);

double GetValidMaxCopyCost();

double GetTransferCost();

void ResizeNdSbpSignature(NdSbpSignature& nd_sbp_sig, int32_t size);

void SetNdSbpSignature(NdSbpSignature* nd_sbp_signature, const SbpSignature& sbp_signature,
                       int32_t sbp_axis);

void DfsGetNdSbpSignature(NdSbpSignature& nd_sbp_sig, int32_t depth, int32_t dims,
                          const SbpSignatureList& sbp_sig_list,
                          std::vector<NdSbpSignature>* nd_sbp_sig_list);

// Compute storage for given NdSbp
double Storage4NdSbp(const NdSbp& nd_sbp, Shape& logical_shape, const Shape& parallel_hierarchy);

// Judge whether an NdSbp could be applied on a tensor with given logical shape
Maybe<bool> FilterNdSbpByLogicalShape(const NdSbp& nd_sbp, Shape& logical_shape,
                                      const Shape& parallel_hierarchy);

// TODO: Unify lazy and eager boxing
Maybe<double> ComputeCopyCostBetweenNdSbp(const NdSbp& producer_sbp_parallel,
                                          const NdSbp& consumer_sbp_parallel,
                                          const BlobDesc& logical_blob_desc,
                                          const ParallelDesc& producer_parallel_desc,
                                          const ParallelDesc& consumer_parallel_desc,
                                          bool requires_same_sbp);

// Cost for boxing in lazy
Maybe<double> ComputeLazyCopyCostBetweenNdSbp(const NdSbp& producer_sbp_parallel,
                                              const NdSbp& consumer_sbp_parallel,
                                              const BlobDesc& logical_blob_desc,
                                              const ParallelDesc& producer_parallel_desc,
                                              const ParallelDesc& consumer_parallel_desc,
                                              bool requires_same_sbp);

// The public interface for computing cost
// It uses the middle nodes algorithm.
Maybe<double> ComputeCopyCostWithMiddleNodes(const NdSbp& producer_sbp_parallel,
                                             const NdSbp& consumer_sbp_parallel,
                                             const BlobDesc& logical_blob_desc,
                                             const ParallelDesc& producer_parallel_desc,
                                             const ParallelDesc& consumer_parallel_desc,
                                             bool requires_same_sbp);

// Decide the priority to infer sbp
// 0: highest priority
// 1.0: normal priority
// 2.0: Penality, the same as infinity
double ComputeSbpInferPriority(const NdSbp& producer_sbp_parallel,
                               const NdSbp& consumer_sbp_parallel,
                               const ParallelDesc& producer_parallel_desc,
                               const ParallelDesc& consumer_parallel_desc, bool requires_same_sbp);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SBP_INFER_UTIL_H_
