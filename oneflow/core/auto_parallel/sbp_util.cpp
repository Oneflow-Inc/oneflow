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

#include <memory>
#include "sbp_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_impl.h"

namespace oneflow {
namespace auto_parallel {

namespace {

// compute copy cost for two SBPs.
// They may be either different or on different devices.
double ComputCopyCostBetweenTwoDiffSbpParallel(const cfg::SbpParallel& producer_sbp_parallel,
                                               const cfg::SbpParallel& consumer_sbp_parallel,
                                               double logical_blob_size, double parallel_num,
                                               bool on_same_devices) {
  // Not supporting S->P for now.
  if (consumer_sbp_parallel.has_partial_sum_parallel()
      && producer_sbp_parallel.has_split_parallel()) {
    return GetMaxVal<float>();
  }
  if (on_same_devices) {
    // B->S, B->P
    if (producer_sbp_parallel.has_broadcast_parallel()) { return 0; }
    // has S
    if (consumer_sbp_parallel.has_split_parallel() || producer_sbp_parallel.has_split_parallel()) {
      if (consumer_sbp_parallel.has_split_parallel()
          && producer_sbp_parallel.has_split_parallel()) {
        // S(0)->S(1), S(1)->S(0), etc.
        return logical_blob_size * (parallel_num - 1) / parallel_num;
      } else {
        // P->S, S->B
        return logical_blob_size * (parallel_num - 1);
      }
    }
    // P->B (= p->S + S->B)
    return 2 * logical_blob_size * (parallel_num - 1);
  } else {
    // They have the same hierarchy at the transfer dimension.
    double overall_cost = logical_blob_size;
    // ? -> B
    if (consumer_sbp_parallel.has_broadcast_parallel()) {
      overall_cost += logical_blob_size * (parallel_num - 1);
    }
    // P -> ?
    if (producer_sbp_parallel.has_partial_sum_parallel()) {
      overall_cost += logical_blob_size * (parallel_num - 1);
    }
    // For B->P, B->S, S->S, overall_cost == logical_blob_size;
    return overall_cost;
  }
}

Maybe<double> ComputCopyCostBetweenTwoNdSbp(const cfg::NdSbp& producer_nd_sbp,
                                            const cfg::NdSbp& consumer_nd_sbp,
                                            double logical_blob_size,
                                            const std::shared_ptr<Shape>& hierarchy,
                                            bool on_same_devices) {
  if (hierarchy->NumAxes() != 2) { return GetMaxVal<float>(); }
  const auto& producer_sbp_size = producer_nd_sbp.sbp_parallel_size();
  const auto& consumer_sbp_size = consumer_nd_sbp.sbp_parallel_size();
  // One of the SBP should have size 2
  CHECK_OR_RETURN((producer_sbp_size == 1 && consumer_sbp_size == 2)
                  || (producer_sbp_size == 2 && consumer_sbp_size == 1)
                  || (producer_sbp_size == 2 && consumer_sbp_size == 2))
      << "Not supporting such boxing type. Check if we have bugs in auto parallel.";
  for (int32_t dim_same_sbp = 0; dim_same_sbp < 2; dim_same_sbp++) {
    // If the nd_sbp only have size 1, then make its dimension 0
    int32_t dim_producer = dim_same_sbp;
    if (producer_sbp_size == 1) { dim_producer = 0; }
    int32_t dim_consumer = dim_same_sbp;
    if (consumer_sbp_size == 1) { dim_consumer = 0; }
    // The SBP parallel are the same at dimension (dim_same_sbp)
    if (producer_nd_sbp.sbp_parallel(dim_producer) == consumer_nd_sbp.sbp_parallel(dim_consumer)) {
      if (!producer_nd_sbp.sbp_parallel(dim_producer).has_split_parallel()) {
        logical_blob_size *= hierarchy->At(dim_same_sbp);
      }
      // The SBP parallel are different at dimension (dim_diff_sbp)
      int32_t dim_diff_sbp = 1 - dim_same_sbp;
      // If the nd_sbp only have size 1, then make its dimension 0.
      // Since we have already do this before, we just maintain the value.
      // Otherwise, switch the dimension to dim_diff_sbp
      if (producer_sbp_size == 2) { dim_producer = dim_diff_sbp; }
      if (consumer_sbp_size == 2) { dim_consumer = dim_diff_sbp; }
      // Spliting at the same dimension needs special cares!
      // Not supported by nccl
      if (dim_diff_sbp == 0
          && producer_nd_sbp.sbp_parallel(dim_producer)
                 != consumer_nd_sbp.sbp_parallel(dim_consumer)
          && (NdSbpAllSameSplitParallel(producer_nd_sbp)
              || NdSbpAllSameSplitParallel(consumer_nd_sbp))) {
        return GetMaxVal<float>();
      }
      return ComputCopyCostBetweenTwoDiffSbpParallel(
          producer_nd_sbp.sbp_parallel(dim_producer), consumer_nd_sbp.sbp_parallel(dim_consumer),
          logical_blob_size, hierarchy->At(dim_diff_sbp), on_same_devices);
    }
  }
  // (1, 2) || (2, 1):
  // Not support something like S0 -> (B, P)
  // (2, 2) :
  // if both dimensions are different, like (S0, S1) -> (S1, S0)
  // TODO: support it recently!
  // TODO: support it recently!
  // TODO: support it recently!
  return GetMaxVal<float>();
}

}  // namespace

// check whether the sbp_parallel is legal
bool CheckSbpParallel(const cfg::SbpParallel& sbp_parallel) {
  // Which checking should we use?
  // return sbp_parallel.parallel_type_case() == SbpParallel::PARALLEL_TYPE_NOT_SET;
  return sbp_parallel.has_split_parallel() || sbp_parallel.has_broadcast_parallel()
         || sbp_parallel.has_partial_sum_parallel();
}

// check whether the nd_sbp is legal
bool CheckNdSbp(const cfg::NdSbp& nd_sbp) {
  if (nd_sbp.sbp_parallel_size() <= 0) { return false; }
  for (const auto& sbp : nd_sbp.sbp_parallel()) {
    if (!CheckSbpParallel(sbp)) { return false; }
  }
  return true;
}

// compute copy cost
Maybe<double> ComputCopyCostBetweenTwoSbpParallel(const cfg::SbpParallel& producer_sbp_parallel,
                                                  const cfg::SbpParallel& consumer_sbp_parallel,
                                                  const BlobDesc& logical_blob_desc,
                                                  const ParallelDesc& producer_parallel_desc,
                                                  const ParallelDesc& consumer_parallel_desc,
                                                  bool is_same_sbp, bool allow_cpu2gpu) {
  // Checking here.
  if (!(CheckSbpParallel(producer_sbp_parallel) && CheckSbpParallel(consumer_sbp_parallel))) {
    return Error::RuntimeError() << "Illegal sbp parallel has been found.";
  }
  // Not supporting S->P for now.
  if (consumer_sbp_parallel.has_partial_sum_parallel()
      && producer_sbp_parallel.has_split_parallel()) {
    return GetMaxVal<float>();
  }
  if (producer_parallel_desc == consumer_parallel_desc) {
    // S->S, B->B, P->P
    if (producer_sbp_parallel == consumer_sbp_parallel) { return 0.0; }
    // Will directly modify output blob of source op. Requiring data having same sbp_parallel
    if (is_same_sbp) { return GetMaxVal<float>(); }
    // B->S, B->P
    if (producer_sbp_parallel.has_broadcast_parallel()) { return 0; }
    double logical_blob_size =
        logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
    // has S
    if (consumer_sbp_parallel.has_split_parallel() || producer_sbp_parallel.has_split_parallel()) {
      if (consumer_sbp_parallel.has_split_parallel()
          && producer_sbp_parallel.has_split_parallel()) {
        // S(0)->S(1), S(1)->S(0), etc.
        return logical_blob_size * (producer_parallel_desc.parallel_num() - 1)
               / producer_parallel_desc.parallel_num();
      } else {
        // P->S, S->B
        return logical_blob_size * (producer_parallel_desc.parallel_num() - 1);
      }
    }
    // P->B (= p->S + S->B)
    return 2 * logical_blob_size * (producer_parallel_desc.parallel_num() - 1);
  } else {
    // Will directly modify output blob of source op. Requiring data having same sbp_parallel
    if (is_same_sbp
        && !(allow_cpu2gpu
             && producer_parallel_desc.EqualsIgnoringDeviceType(consumer_parallel_desc)
             && producer_sbp_parallel == consumer_sbp_parallel)) {
      return GetMaxVal<float>();
    }
    double logical_blob_size =
        logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
    double overall_cost = logical_blob_size;
    // ? -> B
    if (consumer_sbp_parallel.has_broadcast_parallel()) {
      overall_cost += (consumer_parallel_desc.parallel_num() - 1) * logical_blob_size;
    }
    // P -> ?
    if (producer_sbp_parallel.has_partial_sum_parallel()) {
      overall_cost += (producer_parallel_desc.parallel_num() - 1) * logical_blob_size;
    }
    // For B->P, B->S, S->S, overall_cost == logical_blob_size;
    return overall_cost;
  }
}

// compute copy cost
Maybe<double> ComputCopyCostBetweenNdSbp(const cfg::NdSbp& producer_sbp_parallel,
                                         const cfg::NdSbp& consumer_sbp_parallel,
                                         const BlobDesc& logical_blob_desc,
                                         const ParallelDesc& producer_parallel_desc,
                                         const ParallelDesc& consumer_parallel_desc,
                                         bool is_same_sbp, bool allow_cpu2gpu) {
  if (!(CheckNdSbp(producer_sbp_parallel) && CheckNdSbp(consumer_sbp_parallel))) {
    return Error::RuntimeError() << "Illegal sbp parallel has been found.";
  }

  ParallelDesc reduced_in_parallel_desc = producer_parallel_desc;
  ParallelDesc reduced_out_parallel_desc = consumer_parallel_desc;
  cfg::NdSbp reduced_in_nd_sbp;
  cfg::NdSbp reduced_out_nd_sbp;
  InOutParallelDimReduce(producer_parallel_desc, consumer_parallel_desc, producer_sbp_parallel,
                         consumer_sbp_parallel, &reduced_in_parallel_desc,
                         &reduced_out_parallel_desc, &reduced_in_nd_sbp, &reduced_out_nd_sbp);

  const auto& in_hierarchy = reduced_in_parallel_desc.hierarchy();
  const auto& out_hierarchy = reduced_out_parallel_desc.hierarchy();
  int32_t in_dim = in_hierarchy->NumAxes();
  int32_t out_dim = out_hierarchy->NumAxes();
  // Not supporting n-D sbp with n >= 3
  // TODO: Support it in the future
  if (in_dim <= 0 || in_dim >= 3 || out_dim <= 0 || out_dim >= 3) { return GetMaxVal<float>(); }

  bool same_nd_sbp = reduced_in_nd_sbp == reduced_out_nd_sbp;
  if (same_nd_sbp && reduced_in_parallel_desc == reduced_out_parallel_desc) { return 0.0; }
  // Will directly modify output blob of source op. Requiring data having same sbp_parallel
  if (is_same_sbp
      && !(allow_cpu2gpu
           && reduced_in_parallel_desc.EqualsIgnoringDeviceType(reduced_out_parallel_desc)
           && same_nd_sbp)) {
    return GetMaxVal<float>();
  }

  // We support different hierarchy for 1D sbp
  if (in_dim == 1 && out_dim == 1) {
    return ComputCopyCostBetweenTwoSbpParallel(
        reduced_in_nd_sbp.sbp_parallel(0), reduced_out_nd_sbp.sbp_parallel(0), logical_blob_desc,
        reduced_in_parallel_desc, reduced_out_parallel_desc, is_same_sbp, allow_cpu2gpu);
  }
  // Not supporting different hierarchy
  // TODO: Support it in the future
  if (in_hierarchy->elem_cnt() != out_hierarchy->elem_cnt()) { return GetMaxVal<float>(); }

  double logical_blob_size =
      logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
  bool on_same_devices =
      reduced_in_parallel_desc.EqualsIgnoringHierarchy(reduced_out_parallel_desc);

  if (in_dim == 2 && out_dim == 2) {
    // Not supporting different hierarchy
    // TODO: Support it in the future
    if (*in_hierarchy != *out_hierarchy) { return GetMaxVal<float>(); }
    return ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp, logical_blob_size,
                                         in_hierarchy, on_same_devices);
  }

  // (in_dim == 2 && out_dim == 1) || (in_dim == 1 && out_dim == 2)
  if (in_dim == 2 && out_dim == 1) {
    return ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp, logical_blob_size,
                                         in_hierarchy, on_same_devices);
  }

  if (in_dim == 1 && out_dim == 2) {
    return ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp, logical_blob_size,
                                         out_hierarchy, on_same_devices);
  }

  CHECK(false) << "Should not reach here. Something went wrong in ComputCopyCostBetweenNdSbp() in "
                  "sbp_util.cpp.";
  return GetMaxVal<float>();
}

// compute copy cost
Maybe<double> ComputCopyCostBetweenNdSbp(const cfg::NdSbp& producer_sbp_parallel,
                                         const cfg::NdSbp& consumer_sbp_parallel,
                                         double logical_blob_size,
                                         const std::shared_ptr<Shape>& in_hierarchy,
                                         const std::shared_ptr<Shape>& out_hierarchy) {
  if (!(CheckNdSbp(producer_sbp_parallel) && CheckNdSbp(consumer_sbp_parallel))) {
    return Error::RuntimeError() << "Illegal sbp parallel has been found.";
  }

  int32_t in_dim = in_hierarchy->NumAxes();
  int32_t out_dim = out_hierarchy->NumAxes();
  // Not supporting n-D sbp with n >= 3
  // TODO: Support it in the future
  if (in_dim <= 0 || in_dim >= 3 || out_dim <= 0 || out_dim >= 3) { return GetMaxVal<float>(); }

  bool same_nd_sbp = producer_sbp_parallel == consumer_sbp_parallel;
  if (same_nd_sbp && (*in_hierarchy == *out_hierarchy)) { return 0.0; }

  // Not supporting different hierarchy
  // TODO: Support it in the future
  if (in_hierarchy->elem_cnt() != out_hierarchy->elem_cnt()) { return GetMaxVal<float>(); }

  // reduced to 1d sbp
  if (producer_sbp_parallel.sbp_parallel(0) == producer_sbp_parallel.sbp_parallel(1)
      && consumer_sbp_parallel.sbp_parallel(0) == consumer_sbp_parallel.sbp_parallel(1)) {
    return ComputCopyCostBetweenTwoDiffSbpParallel(
        producer_sbp_parallel.sbp_parallel(0), consumer_sbp_parallel.sbp_parallel(1),
        logical_blob_size, in_hierarchy->elem_cnt(), true);
  }

  if (in_dim == 2 && out_dim == 2) {
    // Not supporting different hierarchy
    // TODO: Support it in the future
    if (*in_hierarchy != *out_hierarchy) { return GetMaxVal<float>(); }
    return ComputCopyCostBetweenTwoNdSbp(producer_sbp_parallel, consumer_sbp_parallel,
                                         logical_blob_size, in_hierarchy, true);
  }

  CHECK(false) << "Should not reach here. Something went wrong in ComputCopyCostBetweenNdSbp() in "
                  "sbp_util.cpp.";
  return GetMaxVal<float>();
}

// Judge whether we need the same SBP for both producer and consumer
bool IsSameSbp(OpNode* consumer, const std::string& ibn) {
  // is mutable
  const auto input_blob_modifier_ = consumer->op().InputBlobModifier4Ibn(ibn);
  if (input_blob_modifier_.has_is_mutable() && input_blob_modifier_.is_mutable()) { return true; }
  // kOFRecord or kTensorBuffer don't accept boxing
  const LogicalBlobId& lbi = consumer->op().BnInOp2Lbi(ibn);
  const OpNode& producer = consumer->ProducerOpNode4Lbi(lbi);
  const BlobDesc& logical_blob_desc = producer.LogicalBlobDesc4Lbi(lbi);
  return (logical_blob_desc.data_type() == DataType::kOFRecord
          || logical_blob_desc.data_type() == DataType::kTensorBuffer);
}
}  // namespace auto_parallel
}  // namespace oneflow
