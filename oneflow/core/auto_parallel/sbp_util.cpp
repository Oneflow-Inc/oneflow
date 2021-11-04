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

#include "sbp_util.h"

namespace oneflow {

// check whether the sbp_parallel is legal
bool CheckSbpParallel(const SbpParallel& sbp_parallel) {
  // Which checking should we use?
  // return sbp_parallel.parallel_type_case() == SbpParallel::PARALLEL_TYPE_NOT_SET;
  return sbp_parallel.has_split_parallel() || sbp_parallel.has_broadcast_parallel()
         || sbp_parallel.has_partial_sum_parallel();
}

// compute copy cost
double ComputCopyCostBetweenTwoSbpParallel(const SbpParallel& producer_sbp_parallel,
                                           const SbpParallel& consumer_sbp_parallel,
                                           const BlobDesc& logical_blob_desc,
                                           const ParallelDesc& parallel_desc, bool is_same_sbp) {
  // Checking here.
  if (!(CheckSbpParallel(producer_sbp_parallel) && CheckSbpParallel(consumer_sbp_parallel))) {
    // TODO: replace assert
    std::cout << "Replace assert here!" << std::endl;
    return GetMaxVal<float>();
  }
  // S->S, B->B, P->P
  if (producer_sbp_parallel == consumer_sbp_parallel) { return 0.0; }
  // Will directly modify output blob of source op. Requiring data having same sbp_parallel
  if (is_same_sbp) { return GetMaxVal<float>(); }
  // Not supporting S->P, B->P for now. Actually yes for boxing op, but it does not work with some
  // other ops.
  if (consumer_sbp_parallel.has_partial_sum_parallel()) { return GetMaxVal<float>(); }
  // B->S
  if (producer_sbp_parallel.has_broadcast_parallel()) { return 0; }
  double logical_blob_size =
      logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
  // has S
  if (consumer_sbp_parallel.has_split_parallel() || producer_sbp_parallel.has_split_parallel()) {
    if (consumer_sbp_parallel.has_split_parallel() && producer_sbp_parallel.has_split_parallel()) {
      // S(0)->S(1), S(1)->S(0), etc.
      return logical_blob_size;
    } else {
      // P->S, S->B
      return logical_blob_size * parallel_desc.parallel_num();
    }
  }
  // P->B (= p->S + S->B)
  return 2 * logical_blob_size * parallel_desc.parallel_num();
}

}  // namespace oneflow