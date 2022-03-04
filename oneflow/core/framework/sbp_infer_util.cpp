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

#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/auto_parallel/boxing_collector.h"
#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_impl.h"
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace {

static const double kUnsupportedBoxing = GetMaxVal<float>();

// check whether the sbp_parallel is legal
bool CheckSbpParallel(const SbpParallel& sbp_parallel) {
  return sbp_parallel.has_split_parallel() || sbp_parallel.has_broadcast_parallel()
         || sbp_parallel.has_partial_sum_parallel();
}

// check whether the nd_sbp is legal
bool CheckNdSbp(const NdSbp& nd_sbp) {
  if (nd_sbp.sbp_parallel_size() <= 0) { return false; }
  for (const auto& sbp : nd_sbp.sbp_parallel()) {
    if (!CheckSbpParallel(sbp)) { return false; }
  }
  return true;
}

Maybe<double> ComputCopyCostBetweenTwoSbpParallel(const SbpParallel& producer_sbp_parallel,
                                                  const SbpParallel& consumer_sbp_parallel,
                                                  const BlobDesc& logical_blob_desc,
                                                  const ParallelDesc& producer_parallel_desc,
                                                  const ParallelDesc& consumer_parallel_desc) {
  if (!(CheckSbpParallel(producer_sbp_parallel) && CheckSbpParallel(consumer_sbp_parallel))) {
    return Error::RuntimeError() << "Illegal sbp parallel has been found.";
  }

  // Not supporting S->P for lazy boxing now.
  if (LazyMode::is_enabled()) {
    if (consumer_sbp_parallel.has_partial_sum_parallel()
        && producer_sbp_parallel.has_split_parallel()) {
      return kUnsupportedBoxing;
    }
  }

  if (producer_parallel_desc == consumer_parallel_desc) {
    // Same sbp, no cost: S->S, B->B, P->P
    if (producer_sbp_parallel == consumer_sbp_parallel) { return 0.0; }
    // B->S, B->P
    if (producer_sbp_parallel.has_broadcast_parallel()) { return 1.0; }
    // S->P for eager. It should be 0 as well.
    // NOTE: Similar to B->P, we just make the other part to be 0. You can consider P as S(i) for an
    // arbitrary i.
    if (consumer_sbp_parallel.has_partial_sum_parallel()) { return 1.0; }

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
        // P->S, S->B/P
        return logical_blob_size * (producer_parallel_desc.parallel_num() - 1);
      }
    }
    // P->B
    return 2 * logical_blob_size * (producer_parallel_desc.parallel_num() - 1);
  } else {
    // Not supporting P->P for different placement
    if (LazyMode::is_enabled()) {
      if (consumer_sbp_parallel.has_partial_sum_parallel()
          && producer_sbp_parallel.has_partial_sum_parallel()) {
        return kUnsupportedBoxing;
      }
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

// compute copy cost for two SBPs.
// They may be either different or on different devices.
double ComputCopyCostBetweenTwoDiffSbpParallel(const SbpParallel& producer_sbp_parallel,
                                               const SbpParallel& consumer_sbp_parallel,
                                               double logical_blob_size, double parallel_num,
                                               bool on_same_devices) {
  // Not supporting S->P for now.
  if (consumer_sbp_parallel.has_partial_sum_parallel()
      && producer_sbp_parallel.has_split_parallel()) {
    return kUnsupportedBoxing;
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
    // P->B (= P->S + S->B)
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

Maybe<double> ComputCopyCostBetweenTwoNdSbp(const NdSbp& producer_nd_sbp,
                                            const NdSbp& consumer_nd_sbp, double logical_blob_size,
                                            const std::shared_ptr<Shape>& hierarchy,
                                            bool on_same_devices) {
  if (hierarchy->NumAxes() != 2) { return kUnsupportedBoxing; }
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
        return kUnsupportedBoxing;
      }
      return ComputCopyCostBetweenTwoDiffSbpParallel(
          producer_nd_sbp.sbp_parallel(dim_producer), consumer_nd_sbp.sbp_parallel(dim_consumer),
          logical_blob_size, hierarchy->At(dim_diff_sbp), on_same_devices);
    }
  }
  return kUnsupportedBoxing;
}

Maybe<double> ComputeEagerCopyCostBetweenNdSbp(const NdSbp& producer_sbp_parallel,
                                               const NdSbp& consumer_sbp_parallel,
                                               const BlobDesc& logical_blob_desc,
                                               const ParallelDesc& producer_parallel_desc,
                                               const ParallelDesc& consumer_parallel_desc,
                                               bool requires_same_sbp) {
  if (!(CheckNdSbp(producer_sbp_parallel) && CheckNdSbp(consumer_sbp_parallel))) {
    return Error::RuntimeError() << "Illegal sbp parallel has been found.";
  }

  // TODO: get copy cost from each EagerBoxingInterpreter
  if (!TRY(Global<EagerBoxingInterpreterManager>::Get()->GetEagerBoxingInterpreter(
               producer_sbp_parallel, consumer_sbp_parallel, producer_parallel_desc,
               consumer_parallel_desc, logical_blob_desc.shape()))
           .IsOk()) {
    return kUnsupportedBoxing;
  }

  ParallelDesc reduced_in_parallel_desc = producer_parallel_desc;
  ParallelDesc reduced_out_parallel_desc = consumer_parallel_desc;
  NdSbp reduced_in_nd_sbp;
  NdSbp reduced_out_nd_sbp;
  InOutParallelDimReduce(producer_parallel_desc, consumer_parallel_desc, producer_sbp_parallel,
                         consumer_sbp_parallel, &reduced_in_parallel_desc,
                         &reduced_out_parallel_desc, &reduced_in_nd_sbp, &reduced_out_nd_sbp);

  const auto& in_hierarchy = reduced_in_parallel_desc.hierarchy();
  const auto& out_hierarchy = reduced_out_parallel_desc.hierarchy();

  bool same_nd_sbp = reduced_in_nd_sbp == reduced_out_nd_sbp;
  // Same sbp is always supported.
  if (same_nd_sbp && reduced_in_parallel_desc == reduced_out_parallel_desc) { return 0.0; }
  if (requires_same_sbp) { return kUnsupportedBoxing; }

  int32_t in_dim = in_hierarchy->NumAxes();
  int32_t out_dim = out_hierarchy->NumAxes();
  // We support different hierarchy for 1D sbp
  if (in_dim == 1 && out_dim == 1) {
    return ComputCopyCostBetweenTwoSbpParallel(
        reduced_in_nd_sbp.sbp_parallel(0), reduced_out_nd_sbp.sbp_parallel(0), logical_blob_desc,
        reduced_in_parallel_desc, reduced_out_parallel_desc);
  }

  double total_cost = 0.0;
  if (reduced_in_parallel_desc == reduced_out_parallel_desc) {
    // NOTE: After analysis, transfer cost increase if spliting the same dimension.
    // Example 1: (S(1), S(0), S(1), S(0)) -> (S(0), S(0), S(0), S(0))
    // Example 2: (B, S(0)) -> (S(0), S(0))
    // The cost would be (1-1/n)T, where n is the product of hierarchy number in those splitting
    // dimensions. To give a more precise cost, we add a upper bound of those lost cost back for
    // simplification.
    bool normal_case = true;
    // nd to nd
    for (int32_t i = 0; i < reduced_in_parallel_desc.hierarchy()->NumAxes(); ++i) {
      const auto& in_sbp = reduced_in_nd_sbp.sbp_parallel(i);
      const auto& out_sbp = reduced_out_nd_sbp.sbp_parallel(i);
      // Have bugs here. (B, S0) -> (S0, S0) will give a cost 0.
      // Actually it is (1-1/m)T for hierarchy (n, m)
      // TODO: Fix that after support all sbp combination for eager.
      total_cost += JUST(ComputCopyCostBetweenTwoSbpParallel(
          in_sbp, out_sbp, logical_blob_desc, reduced_in_parallel_desc, reduced_out_parallel_desc));
      // detect the cases that splits the same dimension before this splitting
      if (normal_case && in_sbp.has_split_parallel() && in_sbp == out_sbp) {
        for (int32_t j = 0; j < i; j++) {
          const auto& in_sbp_j = reduced_in_nd_sbp.sbp_parallel(j);
          const auto& out_sbp_j = reduced_out_nd_sbp.sbp_parallel(j);
          // in_sbp == out_sbp in this situation
          if ((in_sbp_j != out_sbp_j) && (in_sbp_j == in_sbp || out_sbp_j == in_sbp)) {
            normal_case = false;
            break;
          }
        }
      }
    }
    // Add the cost for the special case
    if (!normal_case) {
      total_cost +=
          logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
    }
  } else {
    double logical_blob_size =
        logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
    {
      double in_cost = 1.0;
      for (int32_t i = 0; i < reduced_in_parallel_desc.hierarchy()->NumAxes(); ++i) {
        // P -> ?
        if (reduced_in_nd_sbp.sbp_parallel(i).has_partial_sum_parallel()) {
          in_cost *= reduced_in_parallel_desc.hierarchy()->At(i);
        }
      }
      total_cost += logical_blob_size * in_cost;
    }
    {
      double out_cost = 1.0;
      for (int32_t i = 0; i < reduced_out_parallel_desc.hierarchy()->NumAxes(); ++i) {
        // ? -> B
        if (reduced_out_nd_sbp.sbp_parallel(i).has_broadcast_parallel()) {
          out_cost *= reduced_out_parallel_desc.hierarchy()->At(i);
        }
      }
      total_cost += logical_blob_size * out_cost;
    }
  }
  return total_cost;
}

using CopyCostFunc = Maybe<double>(const NdSbp&, const NdSbp&, const BlobDesc&, const ParallelDesc&,
                                   const ParallelDesc&, bool);
Maybe<CopyCostFunc*> GetComputeCopyCostFunc() {
  if (LazyMode::is_enabled()) {
    return &ComputeCopyCostWithMiddleNodes;
  } else {
    return &ComputeEagerCopyCostBetweenNdSbp;
  }
}

}  // namespace

Maybe<double> ComputeLazyCopyCostBetweenNdSbp(const NdSbp& producer_sbp_parallel,
                                              const NdSbp& consumer_sbp_parallel,
                                              const BlobDesc& logical_blob_desc,
                                              const ParallelDesc& producer_parallel_desc,
                                              const ParallelDesc& consumer_parallel_desc,
                                              bool requires_same_sbp) {
  if (!(CheckNdSbp(producer_sbp_parallel) && CheckNdSbp(consumer_sbp_parallel))) {
    return Error::RuntimeError() << "Illegal sbp parallel has been found.";
  }
  ParallelDesc reduced_in_parallel_desc = producer_parallel_desc;
  ParallelDesc reduced_out_parallel_desc = consumer_parallel_desc;
  NdSbp reduced_in_nd_sbp;
  NdSbp reduced_out_nd_sbp;
  InOutParallelDimReduce(producer_parallel_desc, consumer_parallel_desc, producer_sbp_parallel,
                         consumer_sbp_parallel, &reduced_in_parallel_desc,
                         &reduced_out_parallel_desc, &reduced_in_nd_sbp, &reduced_out_nd_sbp);

  const auto& in_hierarchy = reduced_in_parallel_desc.hierarchy();
  const auto& out_hierarchy = reduced_out_parallel_desc.hierarchy();
  int32_t in_dim = in_hierarchy->NumAxes();
  int32_t out_dim = out_hierarchy->NumAxes();
  // Not supporting n-D sbp with n >= 3
  // TODO: Support it in the future
  if (std::min(in_dim, out_dim) <= 0 || std::max(in_dim, out_dim) >= 3) {
    return kUnsupportedBoxing;
  }

  bool same_nd_sbp = reduced_in_nd_sbp == reduced_out_nd_sbp;
  // Same sbp is always supported.
  if (same_nd_sbp && reduced_in_parallel_desc == reduced_out_parallel_desc) { return 0.0; }
  if (requires_same_sbp) { return kUnsupportedBoxing; }

  // We support different hierarchy for 1D sbp
  if (in_dim == 1 && out_dim == 1) {
    return ComputCopyCostBetweenTwoSbpParallel(
        reduced_in_nd_sbp.sbp_parallel(0), reduced_out_nd_sbp.sbp_parallel(0), logical_blob_desc,
        reduced_in_parallel_desc, reduced_out_parallel_desc);
  }
  // Not supporting different hierarchy
  // TODO: Support it in the future
  if (in_hierarchy->elem_cnt() != out_hierarchy->elem_cnt()) { return kUnsupportedBoxing; }

  double logical_blob_size =
      logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
  bool on_same_devices =
      reduced_in_parallel_desc.EqualsIgnoringHierarchy(reduced_out_parallel_desc);

  if (in_dim == 2 && out_dim == 2) {
    // Not supporting different hierarchy
    // TODO: Support it in the future
    if (*in_hierarchy != *out_hierarchy) { return kUnsupportedBoxing; }
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

  return Error::RuntimeError()
         << "Should not reach here. Something went wrong in ComputCopyCostBetweenNdSbp() in "
            "sbp_util.cpp.";
}

double GetValidMaxCopyCost() {
  // We suppose that valid copy cost range is [0, FloatMax*0.8]
  static const double kValidMaxCopyCost = kUnsupportedBoxing * 0.8;
  return kValidMaxCopyCost;
}

void ResizeNdSbpSignature(NdSbpSignature& nd_sbp_sig, int32_t size) {
  for (auto& pair : *nd_sbp_sig.mutable_bn_in_op2nd_sbp()) {
    if (pair.second.sbp_parallel_size() > size) { pair.second.clear_sbp_parallel(); }
    while (pair.second.sbp_parallel_size() < size) { pair.second.add_sbp_parallel(); }
  }
}

void SetNdSbpSignature(NdSbpSignature* nd_sbp_signature, const SbpSignature& sbp_signature,
                       int32_t sbp_axis) {
  for (const auto& pair : sbp_signature.bn_in_op2sbp_parallel()) {
    *((*nd_sbp_signature->mutable_bn_in_op2nd_sbp())[pair.first].mutable_sbp_parallel(sbp_axis)) =
        pair.second;
  }
}

void DfsGetNdSbpSignature(NdSbpSignature& nd_sbp_sig, int32_t depth, int32_t dims,
                          const SbpSignatureList& sbp_sig_list,
                          std::vector<NdSbpSignature>* nd_sbp_sig_list) {
  if (depth == dims) {
    nd_sbp_sig_list->push_back(nd_sbp_sig);
  } else {
    for (const auto& sbp_signature : sbp_sig_list.sbp_signature()) {
      SetNdSbpSignature(&nd_sbp_sig, sbp_signature, depth);
      DfsGetNdSbpSignature(nd_sbp_sig, depth + 1, dims, sbp_sig_list, nd_sbp_sig_list);
    }
  }
}

// Compute storage per device for given NdSbp
double Storage4NdSbp(const NdSbp& nd_sbp, Shape& logical_shape, const Shape& parallel_hierarchy) {
  if (nd_sbp.sbp_parallel_size() == 1) {
    double logical_blob_size = logical_shape.elem_cnt();
    // Checking 1D sbp
    const auto& sbp_parallel = nd_sbp.sbp_parallel(0);
    if (sbp_parallel.has_split_parallel()) {
      const int64_t axis = sbp_parallel.split_parallel().axis();
      if (axis >= logical_shape.NumAxes()) { return kUnsupportedBoxing; }
      if (logical_shape.At(axis) < parallel_hierarchy.At(0)) { return kUnsupportedBoxing; }
      logical_blob_size /= parallel_hierarchy.At(0);
    }
    return logical_blob_size;
  } else {
    for (int32_t dim_sbp = 0; dim_sbp < nd_sbp.sbp_parallel_size(); ++dim_sbp) {
      const auto& sbp_parallel = nd_sbp.sbp_parallel(dim_sbp);
      if (sbp_parallel.has_split_parallel()) {
        // Split axis and store result back to logical shape
        const int64_t axis = sbp_parallel.split_parallel().axis();
        if (axis >= logical_shape.NumAxes()) { return kUnsupportedBoxing; }
        // Use completely average split to count the storage
        if (logical_shape.At(axis) <= 0
            || (logical_shape.At(axis) % parallel_hierarchy.At(dim_sbp) > 0)) {
          return kUnsupportedBoxing;
        }
        logical_shape.Set(axis, logical_shape.At(axis) / parallel_hierarchy.At(dim_sbp));
      }
    }
    return logical_shape.elem_cnt();
  }
}

// Judge whether an NdSbp could be applied on a tensor with given logical shape
// True means this NdSbp is not valid.
Maybe<bool> FilterNdSbpByLogicalShape(const NdSbp& nd_sbp, Shape& logical_shape,
                                      const Shape& parallel_hierarchy) {
  return Storage4NdSbp(nd_sbp, logical_shape, parallel_hierarchy) > GetValidMaxCopyCost();
}

Maybe<double> ComputeCopyCostBetweenNdSbp(const NdSbp& producer_sbp_parallel,
                                          const NdSbp& consumer_sbp_parallel,
                                          const BlobDesc& logical_blob_desc,
                                          const ParallelDesc& producer_parallel_desc,
                                          const ParallelDesc& consumer_parallel_desc,
                                          bool requires_same_sbp) {
  return JUST(GetComputeCopyCostFunc())(producer_sbp_parallel, consumer_sbp_parallel,
                                        logical_blob_desc, producer_parallel_desc,
                                        consumer_parallel_desc, requires_same_sbp);
}

Maybe<double> ComputeCopyCostWithMiddleNodes(const NdSbp& producer_sbp_parallel,
                                             const NdSbp& consumer_sbp_parallel,
                                             const BlobDesc& logical_blob_desc,
                                             const ParallelDesc& producer_parallel_desc,
                                             const ParallelDesc& consumer_parallel_desc,
                                             bool requires_same_sbp) {
  // Initialize boxing collector
  constexpr int32_t kRegularMaxSplitAxes = 6;
  static thread_local BoxingCollector boxing_collector(kRegularMaxSplitAxes);
  std::vector<NdSbp> middle_sbps;
  // Ask for middle nodes
  int32_t diag_node = 0;
  JUST(boxing_collector.AskSbpCombination(
      producer_sbp_parallel, consumer_sbp_parallel, logical_blob_desc, producer_parallel_desc,
      consumer_parallel_desc, /*is_customized=*/false, middle_sbps, &diag_node,
      /*compute_cost=*/true));
  // Parameters
  double total_cost = 0.0;
  double transfer_cost = ParseFloatFromEnv("AUTO_PARALLEL_TRANSFER_COST", 1.65e7);
  // Set up the information of the first node in the first connection
  const NdSbp* pre_nd_sbp = &producer_sbp_parallel;
  const ParallelDesc* pre_parallel_desc = &producer_parallel_desc;
  const ParallelDesc* middle_parallel_desc = nullptr;
  // Connection for the next middle node
  for (int32_t middle_node_id = 0; middle_node_id < middle_sbps.size(); middle_node_id++) {
    const auto& middle_sbp = middle_sbps[middle_node_id];
    if (middle_node_id < diag_node) {
      middle_parallel_desc = &producer_parallel_desc;
    } else {
      middle_parallel_desc = &consumer_parallel_desc;
    }
    // We use the parallel description of consumer as the parallel description for all the middle
    // nodes, following the same procedure in boxing_with_middle_nodes.cpp
    // TODO: Needs more effort if dealing with different placement
    total_cost += JUST(ComputeLazyCopyCostBetweenNdSbp(*pre_nd_sbp, middle_sbp, logical_blob_desc,
                                                       *pre_parallel_desc, *middle_parallel_desc,
                                                       requires_same_sbp))
                  + transfer_cost;
    // Set up the information of the first node in the next connection
    pre_nd_sbp = &middle_sbp;
    pre_parallel_desc = middle_parallel_desc;
  }
  // Connection between the last middle node and consumer
  total_cost += JUST(ComputeLazyCopyCostBetweenNdSbp(*pre_nd_sbp, consumer_sbp_parallel,
                                                     logical_blob_desc, *pre_parallel_desc,
                                                     consumer_parallel_desc, requires_same_sbp));

  return total_cost;
}

}  // namespace oneflow
