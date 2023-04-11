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

// [2, 3, 4, 5, 9, 100, 8]: (P, S0, P, P, B, S1, P)
// partial ratio = 2 * 4 * 5 * 8
int32_t PartialRatio4Producer(const NdSbp& sbp_producer,
                              const ParallelDesc& producer_parallel_desc);

// [2, 3, 4, 5, 9, 100, 8]: (P, S0, B, P, B, S1, P)
// broadcast ratio = 4 * 9
int32_t BroadcastRatio4Consumer(const NdSbp& sbp_consumer,
                                const ParallelDesc& consumer_parallel_desc);

void NdSbpDimReduce(const Shape& hierarchy, const NdSbp& nd_sbp, Shape* reduced_hierarchy,
                    NdSbp* reduced_nd_sbp, const Shape& logical_shape);
void NdSbpsDimReduce(const Shape& hierarchy, const std::vector<const NdSbp*>& nd_sbps,
                     Shape* reduced_hierarchy, const std::vector<NdSbp*>& reduced_nd_sbps,
                     const Shape& logical_shape);
void NdSbpDimReduce(const ParallelDesc& parallel_desc, const NdSbp& nd_sbp,
                    ParallelDesc* reduced_parallel_desc, NdSbp* reduced_nd_sbp,
                    const Shape& logical_shape);

void InOutParallelDimReduce(const Shape& in_hierarchy, const Shape& out_hierarchy,
                            const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
                            Shape* reduced_in_hierarchy, Shape* reduced_out_hierarchy,
                            NdSbp* reduced_in_nd_sbp, NdSbp* reduced_out_nd_sbp,
                            const Shape& logical_shape);
void InOutParallelDimReduce(const ParallelDesc& in_parallel_desc,
                            const ParallelDesc& out_parallel_desc, const NdSbp& in_nd_sbp,
                            const NdSbp& out_nd_sbp, ParallelDesc* reduced_in_parallel_desc,
                            ParallelDesc* reduced_out_parallel_desc, NdSbp* reduced_in_nd_sbp,
                            NdSbp* reduced_out_nd_sbp, const Shape& logical_shape);

double GetValidMaxCopyCost();

double GetTransferCost();

void ResizeNdSbpSignature(NdSbpSignature& nd_sbp_sig, int32_t size);

void SetNdSbpSignature(NdSbpSignature* nd_sbp_signature, const SbpSignature& sbp_signature,
                       int32_t sbp_axis);

void DfsGetNdSbpSignature(NdSbpSignature& nd_sbp_sig, int32_t depth, int32_t dims,
                          const Shape& hierarchy,
                          const HashMap<int32_t, SbpSignatureList>& hierarchy_value2sbp_sig_list,
                          std::vector<NdSbpSignature>* nd_sbp_sig_list);

void DeduplicateNdSbpSignatureList(std::vector<NdSbpSignature>* nd_sbp_sig_list,
                                   const std::vector<std::string>& bns);

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
// 2.0: Penalty, the same as infinity
double ComputeSbpInferPriority(const NdSbp& producer_sbp_parallel,
                               const NdSbp& consumer_sbp_parallel,
                               const ParallelDesc& producer_parallel_desc,
                               const ParallelDesc& consumer_parallel_desc, bool requires_same_sbp,
                               const Shape& logical_shape);

// The transfer ratio for general basic communication
// Cost = ratio * data amount
double Cost4GeneralBasicCommunication(const NdSbp& producer_sbp_parallel,
                                      const NdSbp& consumer_sbp_parallel,
                                      const BlobDesc& logical_blob_desc,
                                      const ParallelDesc& producer_parallel_desc,
                                      const ParallelDesc& consumer_parallel_desc);

int64_t TotalByteSize4BlobDesc(const BlobDesc& logical_blob_desc);
int64_t MaxByteSize4BlobDescSbp(const BlobDesc& logical_blob_desc, const NdSbp& nd_sbp,
                                const Shape& hierarchy);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SBP_INFER_UTIL_H_
