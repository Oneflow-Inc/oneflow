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
#include "oneflow/core/auto_parallel/algorithm_util.h"
#include "oneflow/core/auto_parallel/boxing_collector.h"
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/register/blob_desc.h"

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

double Penalty4PartialInConsumer(double logical_blob_size, int32_t producer_parallel_num,
                                 int32_t consumer_parallel_num) {
  static const int64_t penalty4partial_in_consumer_tag =
      ParseIntegerFromEnv("ONEFLOW_PENALTY_FOR_PARTIAL_IN_CONSUMER_POLICY", 2);
  if (penalty4partial_in_consumer_tag == Penalty4PartialInConsumerTag::kSlight) {
    return 1.0;
  } else if (penalty4partial_in_consumer_tag == Penalty4PartialInConsumerTag::kMiddle) {
    return 4 * logical_blob_size * (producer_parallel_num + consumer_parallel_num);
  } else {
    return kUnsupportedBoxing;
  }
}

int32_t Ratio4Sbp(const NdSbp& nd_sbp, const ParallelDesc& parallel_desc,
                  const std::function<bool(const SbpParallel&)>& classifier) {
  int32_t ratio = 1;
  for (int32_t sbp_id = 0; sbp_id < nd_sbp.sbp_parallel_size(); sbp_id++) {
    if (classifier(nd_sbp.sbp_parallel(sbp_id))) { ratio *= parallel_desc.hierarchy()->At(sbp_id); }
  }
  return ratio;
}

Maybe<double> ComputCopyCostBetweenTwoSbpParallel(const SbpParallel& producer_sbp_parallel,
                                                  const SbpParallel& consumer_sbp_parallel,
                                                  const BlobDesc& logical_blob_desc,
                                                  bool on_same_devices,
                                                  int32_t producer_parallel_num,
                                                  int32_t consumer_parallel_num) {
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

  // NOTE: A tensor placed on cpu with a consumer operator that accepts cuda inputs would be
  // transferred to cuda later. We might not have correct parallel description at this moment.
  if (on_same_devices && producer_parallel_num == consumer_parallel_num) {
    // Same sbp, no cost: S->S, B->B, P->P
    if (producer_sbp_parallel == consumer_sbp_parallel) { return 0.0; }
    double logical_blob_size = TotalByteSize4BlobDesc(logical_blob_desc);
    // S->P for eager. It should be 0 as well.
    // NOTE: Similar to B->P, we just make the other part to be 0. You can consider P as S(i) for an
    // arbitrary i.
    // ? -> P
    if (consumer_sbp_parallel.has_partial_sum_parallel()) {
      return Penalty4PartialInConsumer(logical_blob_size, producer_parallel_num,
                                       consumer_parallel_num);
    }
    // B->S
    if (producer_sbp_parallel.has_broadcast_parallel()) { return 1.0; }

    // has S
    if (consumer_sbp_parallel.has_split_parallel() || producer_sbp_parallel.has_split_parallel()) {
      if (consumer_sbp_parallel.has_split_parallel()
          && producer_sbp_parallel.has_split_parallel()) {
        // S(0)->S(1), S(1)->S(0), etc.
        return logical_blob_size * (producer_parallel_num - 1) / producer_parallel_num;
      } else {
        // P->S, S->B/P
        return logical_blob_size * (producer_parallel_num - 1);
      }
    }
    // P->B
    return 2 * logical_blob_size * (producer_parallel_num - 1);
  } else {
    // Not supporting P->P for different placement
    if (LazyMode::is_enabled()) {
      if (consumer_sbp_parallel.has_partial_sum_parallel()
          && producer_sbp_parallel.has_partial_sum_parallel()) {
        return kUnsupportedBoxing;
      }
    }

    double logical_blob_size = TotalByteSize4BlobDesc(logical_blob_desc);
    double overall_cost = logical_blob_size;
    // ? -> B
    if (consumer_sbp_parallel.has_broadcast_parallel()) {
      overall_cost += (consumer_parallel_num - 1) * logical_blob_size;
    }
    // P -> ?
    if (producer_sbp_parallel.has_partial_sum_parallel()) {
      overall_cost += (producer_parallel_num - 1) * logical_blob_size;
    }
    // ? -> P
    if (consumer_sbp_parallel.has_partial_sum_parallel()) {
      overall_cost += Penalty4PartialInConsumer(logical_blob_size, producer_parallel_num,
                                                consumer_parallel_num);
    }
    // For B->S, S->S, overall_cost == logical_blob_size;
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
    // B->P
    if (consumer_sbp_parallel.has_partial_sum_parallel()) {
      return Penalty4PartialInConsumer(logical_blob_size, parallel_num, parallel_num);
    }
    // B->S
    if (producer_sbp_parallel.has_broadcast_parallel()) { return 1; }
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
    if (consumer_sbp_parallel.has_partial_sum_parallel()) {
      overall_cost += Penalty4PartialInConsumer(logical_blob_size, parallel_num, parallel_num);
    }
    // For B->P, B->S, S->S, overall_cost == logical_blob_size;
    return overall_cost;
  }
}

Maybe<double> ComputCopyCostBetweenTwoNdSbp(const NdSbp& producer_nd_sbp,
                                            const NdSbp& consumer_nd_sbp, double logical_blob_size,
                                            const Shape& hierarchy, bool on_same_devices) {
  if (hierarchy.NumAxes() != 2) { return kUnsupportedBoxing; }
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
        logical_blob_size *= hierarchy.At(dim_same_sbp);
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
          logical_blob_size, hierarchy.At(dim_diff_sbp), on_same_devices);
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
  if (!TRY(Singleton<EagerBoxingInterpreterManager>::Get()->GetEagerBoxingInterpreter(
               producer_sbp_parallel, consumer_sbp_parallel, producer_parallel_desc,
               consumer_parallel_desc, logical_blob_desc.shape()))
           .IsOk()) {
    return kUnsupportedBoxing;
  }

  bool on_same_devices = producer_parallel_desc.EqualsIgnoringHierarchy(consumer_parallel_desc);

  // Reduce before cost computation
  Shape reduced_in_hierarchy;
  NdSbp reduced_in_nd_sbp;
  Shape reduced_out_hierarchy;
  NdSbp reduced_out_nd_sbp;
  InOutParallelDimReduce(*producer_parallel_desc.hierarchy(), *consumer_parallel_desc.hierarchy(),
                         producer_sbp_parallel, consumer_sbp_parallel, &reduced_in_hierarchy,
                         &reduced_out_hierarchy, &reduced_in_nd_sbp, &reduced_out_nd_sbp,
                         logical_blob_desc.shape());

  bool same_nd_sbp = reduced_in_nd_sbp == reduced_out_nd_sbp;
  // Same sbp is always supported.
  if (same_nd_sbp && on_same_devices && reduced_in_hierarchy == reduced_out_hierarchy) {
    return 0.0;
  }
  if (requires_same_sbp) { return kUnsupportedBoxing; }

  int32_t in_dim = reduced_in_hierarchy.NumAxes();
  int32_t out_dim = reduced_out_hierarchy.NumAxes();
  // We support different hierarchy for 1D sbp
  if (in_dim == 1 && out_dim == 1) {
    return ComputCopyCostBetweenTwoSbpParallel(
        reduced_in_nd_sbp.sbp_parallel(0), reduced_out_nd_sbp.sbp_parallel(0), logical_blob_desc,
        on_same_devices, reduced_in_hierarchy.elem_cnt(), reduced_out_hierarchy.elem_cnt());
  }

  double total_cost = 1.0;
  if (on_same_devices && reduced_in_hierarchy == reduced_out_hierarchy) {
    // NOTE: After analysis, transfer cost increase if spliting the same dimension.
    // Example 1: (S(1), S(0), S(1), S(0)) -> (S(0), S(0), S(0), S(0))
    // Example 2: (B, S(0)) -> (S(0), S(0))
    // The cost would be (1-1/n)T, where n is the product of hierarchy number in those splitting
    // dimensions. To give a more precise cost, we add a upper bound of those lost cost back for
    // simplification.
    bool normal_case = true;
    // nd to nd
    for (int32_t i = 0; i < in_dim; ++i) {
      const auto& in_sbp = reduced_in_nd_sbp.sbp_parallel(i);
      const auto& out_sbp = reduced_out_nd_sbp.sbp_parallel(i);
      // Have bugs here. (B, S0) -> (S0, S0) will give a cost 0.
      // Actually it is (1-1/m)T for hierarchy (n, m)
      // TODO: Fix that after support all sbp combination for eager.
      total_cost += JUST(ComputCopyCostBetweenTwoSbpParallel(
          in_sbp, out_sbp, logical_blob_desc, on_same_devices, reduced_in_hierarchy.elem_cnt(),
          reduced_out_hierarchy.elem_cnt()));
      // Add the penalty for P in the consumer
      if (out_sbp.has_partial_sum_parallel() && (in_sbp != out_sbp)) {
        total_cost += Penalty4PartialInConsumer(TotalByteSize4BlobDesc(logical_blob_desc),
                                                producer_parallel_desc.parallel_num(),
                                                consumer_parallel_desc.parallel_num());
      }
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
    if (!normal_case) { total_cost += TotalByteSize4BlobDesc(logical_blob_desc); }
  } else {
    double logical_blob_size = TotalByteSize4BlobDesc(logical_blob_desc);
    {
      double in_cost = 1.0;
      for (int32_t i = 0; i < in_dim; ++i) {
        // P -> ?
        if (reduced_in_nd_sbp.sbp_parallel(i).has_partial_sum_parallel()) {
          in_cost *= reduced_in_hierarchy.At(i);
        }
      }
      total_cost += logical_blob_size * in_cost;
    }
    {
      double out_cost = 1.0;
      for (int32_t i = 0; i < out_dim; ++i) {
        // ? -> B
        if (reduced_out_nd_sbp.sbp_parallel(i).has_broadcast_parallel()) {
          out_cost *= reduced_out_hierarchy.At(i);
        }
        // Add the penalty for P in the consumer
        if (reduced_out_nd_sbp.sbp_parallel(i).has_partial_sum_parallel()) {
          total_cost +=
              Penalty4PartialInConsumer(logical_blob_size, producer_parallel_desc.parallel_num(),
                                        consumer_parallel_desc.parallel_num());
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

// Replace the hierarchy and then create a new parallel description
void ReplaceHierarchy4ParallelDesc(const ParallelDesc& old_parallel_desc,
                                   const Shape& new_hierarchy, ParallelDesc* new_parallel_desc) {
  if (*old_parallel_desc.hierarchy() == new_hierarchy) {
    *new_parallel_desc = old_parallel_desc;
  } else {
    ParallelConf new_parallel_conf = old_parallel_desc.parallel_conf();
    new_hierarchy.ToProto(new_parallel_conf.mutable_hierarchy());
    *new_parallel_desc = ParallelDesc(new_parallel_conf);
  }
}

// We can not just simply merging two same split
// For example, shape = [6], we are trying to merge [2, 2]: (S0, S0) -> [4]: S0
// For each rank, [4]: S0 has number of data: 2, 2, 1, 1
// For each rank, [2]: S0 has number of data: 3, 3
// For each rank, [2, 2]: (S0, S0) has number of data: 2, 1, 2, 1
// Thus {[2, 2]: (S0, S0)} != {[4]: S0} for shape [6]
// However {[2, 2]: (S0, S0)} == {[4]: S0} for shape [4], [5], [7], [8]
// More specifically, {[a, b]: (Si, Si)} == {[a*b]: Si} if and only if
// shape value % (a * b) == 0, 1, a*b - 1
bool CanMergeSplit(int32_t shape_value, int32_t merged_split_hierarchy_value) {
  int32_t remainder = shape_value % merged_split_hierarchy_value;
  if (remainder <= 1 || remainder == merged_split_hierarchy_value - 1) {
    return true;
  } else {
    return false;
  }
}

}  // namespace

int32_t PartialRatio4Producer(const NdSbp& sbp_producer,
                              const ParallelDesc& producer_parallel_desc) {
  return Ratio4Sbp(sbp_producer, producer_parallel_desc, &SbpParallel::has_partial_sum_parallel);
}

int32_t BroadcastRatio4Consumer(const NdSbp& sbp_consumer,
                                const ParallelDesc& consumer_parallel_desc) {
  return Ratio4Sbp(sbp_consumer, consumer_parallel_desc, &SbpParallel::has_broadcast_parallel);
}

void NdSbpDimReduce(const Shape& hierarchy, const NdSbp& nd_sbp, Shape* reduced_hierarchy,
                    NdSbp* reduced_nd_sbp, const Shape& logical_shape) {
  NdSbpsDimReduce(hierarchy, {&nd_sbp}, reduced_hierarchy, {reduced_nd_sbp}, logical_shape);
}

void NdSbpsDimReduce(const Shape& hierarchy, const std::vector<const NdSbp*>& nd_sbps,
                     Shape* reduced_hierarchy, const std::vector<NdSbp*>& reduced_nd_sbps,
                     const Shape& logical_shape) {
  int32_t sbp_num = nd_sbps.size();
  // Speed up for 1d sbp
  if (hierarchy.NumAxes() == 1) {
    *reduced_hierarchy = hierarchy;
    for (int32_t index = 0; index < sbp_num; index++) {
      if (hierarchy.elem_cnt() == 1) {
        reduced_nd_sbps[index]->add_sbp_parallel()->mutable_broadcast_parallel();
      } else {
        *reduced_nd_sbps[index] = *nd_sbps[index];
      }
    }
    return;
  }
  reduced_hierarchy->clear();
  for (auto& reduced_nd_sbp : reduced_nd_sbps) { reduced_nd_sbp->clear_sbp_parallel(); }
  // At this moment, if we have [2, 4, 3, 7]: (S0, S1, S0, S0) for logical shape [601, 301, 999]
  // We hold the split when accessing the current dimension
  // Do the true splitting until we reach the next step
  // dim = 0, split_axis2holding_reduced_shapes: {(0: 601)}, last split axis = -1
  // dim = 1, split_axis2holding_reduced_shapes: {(0: 300, 301), (1: 301)}, last split axis = 0
  // dim = 2, split_axis2holding_reduced_shapes: {(0: 300, 301), (1: 75, 76)}, last split axis = 1
  // dim = 3, at this moment, last split axis (0) == current split axis (0),
  // dim = 3, but judging 300 % (3 * 7) = 6 fails the CanMergeSplit(), not merging
  // dim = 3, split_axis2holding_reduced_shapes: {(0: 100, 101), (1: 75, 76)}, last split axis = 0
  std::vector<HashMap<int32_t, HashSet<int32_t>>> index2split_axis2holding_reduced_shapes(sbp_num);
  std::vector<std::vector<int32_t>> index2last_holding_reduced_shapes(sbp_num);
  std::vector<int32_t> last_split_axises(sbp_num, -1);
  std::vector<int32_t> indexes(sbp_num);
  for (int32_t index = 0; index < sbp_num; index++) { indexes[index] = index; }
  auto add_to_reduced_sbp_hierarchy = [&](int32_t hierarchy_dim) {
    // Clear the last holding split axis
    for (int32_t index = 0; index < sbp_num; index++) {
      auto& split_axis2holding_reduced_shapes = index2split_axis2holding_reduced_shapes[index];
      auto& last_holding_reduced_shapes = index2last_holding_reduced_shapes[index];
      auto& last_split_axis = last_split_axises[index];
      auto& nd_sbp = nd_sbps[index];
      auto& reduced_nd_sbp = reduced_nd_sbps[index];
      if (last_split_axis >= 0) {
        auto& holding_reduced_shapes = split_axis2holding_reduced_shapes[last_split_axis];
        holding_reduced_shapes.clear();
        for (int32_t last_holding_reduced_shape : last_holding_reduced_shapes) {
          int32_t quotient = last_holding_reduced_shape / reduced_hierarchy->back();
          if (last_holding_reduced_shape % reduced_hierarchy->back() != 0) {
            holding_reduced_shapes.insert(quotient + 1);
          }
          holding_reduced_shapes.insert(quotient);
        }
      }
      // Add a new sbp_parallel and a new hierarchy dimension
      const auto& curr_sbp_parallel = nd_sbp->sbp_parallel(hierarchy_dim);
      *reduced_nd_sbp->add_sbp_parallel() = curr_sbp_parallel;
      // Hold the current split shape
      if (curr_sbp_parallel.has_split_parallel()) {
        last_holding_reduced_shapes.clear();
        last_split_axis = curr_sbp_parallel.split_parallel().axis();
        auto it = split_axis2holding_reduced_shapes.find(last_split_axis);
        if (it == split_axis2holding_reduced_shapes.end()) {
          // Looking at a dimension which is never splitted before
          // Shape: [601, ...], sbp: (S0, ...)
          last_holding_reduced_shapes.push_back(logical_shape.At(last_split_axis));
        } else {
          // This dimension is splitted before
          // Shape: [601, 301, ...], sbp: (S0, S1, B, S0, ...), hierarchy: [2, 3, 100, 7, ...]
          // Looking at i = 3, we hold the second S0, but 601 is already splitted by the first S0.
          // split_axis2holding_reduced_shapes: {(0: 300, 301), (1: 100, 101)}
          last_holding_reduced_shapes.assign(it->second.begin(), it->second.end());
        }
      } else {
        last_split_axis = -1;
      }
    }
    // Add a new hierarchy dimension
    reduced_hierarchy->emplace_back(hierarchy.At(hierarchy_dim));
  };
  for (int32_t hierarchy_dim = 0; hierarchy_dim < hierarchy.NumAxes(); hierarchy_dim++) {
    // Shrink those dimension with hierarchy value = 1
    if (hierarchy.At(hierarchy_dim) == 1) { continue; }
    if (reduced_hierarchy->empty()) {
      // Empty hierarchy, add to the back
      add_to_reduced_sbp_hierarchy(hierarchy_dim);
      continue;
    }
    if (std::all_of(indexes.begin(), indexes.end(), [&](int32_t index) {
          // reduced_hierarchy->size() == reduced_nd_sbps[index]->sbp_parallel_size()
          // Basically, current nd sbp == reduced nd sbp.back()
          return nd_sbps[index]->sbp_parallel(hierarchy_dim)
                 == reduced_nd_sbps[index]->sbp_parallel(reduced_hierarchy->size() - 1);
        })) {
      int32_t merged_hierarchy_value = reduced_hierarchy->back() * hierarchy.At(hierarchy_dim);
      // You can merge two sbp with B or P.
      // If sbp = S, then you need to make sure that all the shape value can be splitted
      if (std::all_of(indexes.begin(), indexes.end(), [&](int32_t index) {
            return !nd_sbps[index]->sbp_parallel(hierarchy_dim).has_split_parallel()
                   || std::all_of(index2last_holding_reduced_shapes[index].begin(),
                                  index2last_holding_reduced_shapes[index].end(), [&](int32_t i) {
                                    return CanMergeSplit(i, merged_hierarchy_value);
                                  });
          })) {
        // Merge sbp and hierarchy
        reduced_hierarchy->back() = merged_hierarchy_value;
        continue;
      }
    }
    // Can not merge, add to the back
    add_to_reduced_sbp_hierarchy(hierarchy_dim);
  }
  // [1, 1, ..., 1]: Any --> [1]: (B)
  if (reduced_hierarchy->empty()) {
    reduced_hierarchy->emplace_back(hierarchy.At(0));
    for (auto& reduced_nd_sbp : reduced_nd_sbps) {
      reduced_nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
    }
  }
}

void NdSbpDimReduce(const ParallelDesc& parallel_desc, const NdSbp& nd_sbp,
                    ParallelDesc* reduced_parallel_desc, NdSbp* reduced_nd_sbp,
                    const Shape& logical_shape) {
  // Speed up for 1d sbp
  if (parallel_desc.hierarchy()->NumAxes() == 1) {
    *reduced_parallel_desc = parallel_desc;
    *reduced_nd_sbp = nd_sbp;
    return;
  }
  Shape reduced_hierarchy;
  NdSbpDimReduce(*parallel_desc.hierarchy(), nd_sbp, &reduced_hierarchy, reduced_nd_sbp,
                 logical_shape);

  ReplaceHierarchy4ParallelDesc(parallel_desc, reduced_hierarchy, reduced_parallel_desc);
}

void InOutParallelDimReduce(const Shape& in_hierarchy, const Shape& out_hierarchy,
                            const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
                            Shape* reduced_in_hierarchy, Shape* reduced_out_hierarchy,
                            NdSbp* reduced_in_nd_sbp, NdSbp* reduced_out_nd_sbp,
                            const Shape& logical_shape) {
  if (in_hierarchy == out_hierarchy) {
    // [2, 4]: (S0, S0) -> [2, 4]: (S0, S1)
    NdSbpsDimReduce(in_hierarchy, {&in_nd_sbp, &out_nd_sbp}, reduced_in_hierarchy,
                    {reduced_in_nd_sbp, reduced_out_nd_sbp}, logical_shape);
    *reduced_out_hierarchy = *reduced_in_hierarchy;
  } else {
    // [2, 4]: (S0, S0) -> [4, 2]: (S0, S1)
    // [2, 4]: (S0, S0) -> [3, 3]: (S0, S1)
    NdSbpDimReduce(in_hierarchy, in_nd_sbp, reduced_in_hierarchy, reduced_in_nd_sbp, logical_shape);
    NdSbpDimReduce(out_hierarchy, out_nd_sbp, reduced_out_hierarchy, reduced_out_nd_sbp,
                   logical_shape);

    // Sbp of 3d or higher dimension would use general basic communication
    // Only looks at 1d to 2d or 2d to 1d
    if (reduced_in_hierarchy->NumAxes() + reduced_out_hierarchy->NumAxes() == 3
        && reduced_in_hierarchy->elem_cnt() == reduced_out_hierarchy->elem_cnt()) {
      if (reduced_in_hierarchy->NumAxes() == 1) {
        // [8]: S0 -> [4, 2]: (S0, S1)
        // [8]: B -> [2, 4]: (S0, S1)
        const auto& in_sbp_parallel = reduced_in_nd_sbp->sbp_parallel(0);
        if (!in_sbp_parallel.has_split_parallel()
            || CanMergeSplit(logical_shape.At(in_sbp_parallel.split_parallel().axis()),
                             reduced_in_hierarchy->elem_cnt())) {
          // Change [8]: S0 -> [4, 2]: (S0, S1) to [4, 2]: (S0, S0) -> [4, 2]: (S0, S1)
          // Change [8]: B -> [2, 4]: (S0, S1) to [2, 4]: (B, B) -> [2, 4]: (S0, S1)
          *reduced_in_nd_sbp->add_sbp_parallel() = in_sbp_parallel;
          *reduced_in_hierarchy = *reduced_out_hierarchy;
        }
      } else {
        // [2, 3]: (S0, P) -> [6]: S0
        // [3, 4]: (B, S1) -> [12]: B
        const auto& out_sbp_parallel = reduced_out_nd_sbp->sbp_parallel(0);
        if (!out_sbp_parallel.has_split_parallel()
            || CanMergeSplit(logical_shape.At(out_sbp_parallel.split_parallel().axis()),
                             reduced_out_hierarchy->elem_cnt())) {
          // Change [2, 3]: (S0, P) -> [6]: S0 to [2, 3]: (S0, P) -> [2, 3]: (S0, S0)
          // Change [3, 4]: (B, S1) -> [12]: B to [3, 4]: (B, S1) -> [3, 4]: (B, B)
          *reduced_out_nd_sbp->add_sbp_parallel() = out_sbp_parallel;
          *reduced_out_hierarchy = *reduced_in_hierarchy;
        }
      }
    }
  }
}

void InOutParallelDimReduce(const ParallelDesc& in_parallel_desc,
                            const ParallelDesc& out_parallel_desc, const NdSbp& in_nd_sbp,
                            const NdSbp& out_nd_sbp, ParallelDesc* reduced_in_parallel_desc,
                            ParallelDesc* reduced_out_parallel_desc, NdSbp* reduced_in_nd_sbp,
                            NdSbp* reduced_out_nd_sbp, const Shape& logical_shape) {
  // Speed up for 1d sbp
  if (in_parallel_desc.hierarchy()->NumAxes() == 1
      && out_parallel_desc.hierarchy()->NumAxes() == 1) {
    *reduced_in_parallel_desc = in_parallel_desc;
    *reduced_out_parallel_desc = out_parallel_desc;
    *reduced_in_nd_sbp = in_nd_sbp;
    *reduced_out_nd_sbp = out_nd_sbp;
  } else {
    Shape reduced_in_hierarchy;
    Shape reduced_out_hierarchy;
    InOutParallelDimReduce(*in_parallel_desc.hierarchy(), *out_parallel_desc.hierarchy(), in_nd_sbp,
                           out_nd_sbp, &reduced_in_hierarchy, &reduced_out_hierarchy,
                           reduced_in_nd_sbp, reduced_out_nd_sbp, logical_shape);
    ReplaceHierarchy4ParallelDesc(in_parallel_desc, reduced_in_hierarchy, reduced_in_parallel_desc);
    ReplaceHierarchy4ParallelDesc(out_parallel_desc, reduced_out_hierarchy,
                                  reduced_out_parallel_desc);
  }
}

int64_t TotalByteSize4BlobDesc(const BlobDesc& logical_blob_desc) {
  return logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
}

int64_t MaxByteSize4BlobDescSbp(const BlobDesc& logical_blob_desc, const NdSbp& nd_sbp,
                                const Shape& hierarchy) {
  Shape blob_shape = logical_blob_desc.shape();
  for (int32_t sbp_id = 0; sbp_id < nd_sbp.sbp_parallel_size(); sbp_id++) {
    const auto& sbp = nd_sbp.sbp_parallel(sbp_id);
    if (sbp.has_split_parallel()) {
      int32_t split_axis = sbp.split_parallel().axis();
      blob_shape.Set(split_axis, CeilQuotient(blob_shape.At(split_axis), hierarchy.At(sbp_id)));
    }
  }
  return blob_shape.elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
}

Maybe<double> ComputeLazyCopyCostBetweenNdSbp(const NdSbp& producer_sbp_parallel,
                                              const NdSbp& consumer_sbp_parallel,
                                              const BlobDesc& logical_blob_desc,
                                              const ParallelDesc& producer_parallel_desc,
                                              const ParallelDesc& consumer_parallel_desc,
                                              bool requires_same_sbp) {
  if (!(CheckNdSbp(producer_sbp_parallel) && CheckNdSbp(consumer_sbp_parallel))) {
    return Error::RuntimeError() << "Illegal sbp parallel has been found.";
  }
  bool on_same_devices = producer_parallel_desc.EqualsIgnoringHierarchy(consumer_parallel_desc);

  // Reduce before cost computation
  Shape reduced_in_hierarchy;
  NdSbp reduced_in_nd_sbp;
  Shape reduced_out_hierarchy;
  NdSbp reduced_out_nd_sbp;
  InOutParallelDimReduce(*producer_parallel_desc.hierarchy(), *consumer_parallel_desc.hierarchy(),
                         producer_sbp_parallel, consumer_sbp_parallel, &reduced_in_hierarchy,
                         &reduced_out_hierarchy, &reduced_in_nd_sbp, &reduced_out_nd_sbp,
                         logical_blob_desc.shape());
  int32_t in_dim = reduced_in_hierarchy.NumAxes();
  int32_t out_dim = reduced_out_hierarchy.NumAxes();
  // Not supporting n-D sbp with n >= 3
  // TODO: Support it in the future
  if (std::min(in_dim, out_dim) <= 0 || std::max(in_dim, out_dim) >= 3) {
    return kUnsupportedBoxing;
  }

  bool same_nd_sbp = reduced_in_nd_sbp == reduced_out_nd_sbp;
  // Same sbp is always supported.
  if (same_nd_sbp && on_same_devices && reduced_in_hierarchy == reduced_out_hierarchy) {
    return 0.0;
  }
  if (requires_same_sbp) { return kUnsupportedBoxing; }

  // We support different hierarchy for 1D sbp
  if (in_dim == 1 && out_dim == 1) {
    return GetTransferCost()
           + JUST(ComputCopyCostBetweenTwoSbpParallel(
               reduced_in_nd_sbp.sbp_parallel(0), reduced_out_nd_sbp.sbp_parallel(0),
               logical_blob_desc, on_same_devices, reduced_in_hierarchy.elem_cnt(),
               reduced_out_hierarchy.elem_cnt()));
  }

#ifdef WITH_CUDA
  static const bool enable_general_basic_communication =
      ParseBooleanFromEnv("ONEFLOW_BOXING_ENABLE_GENERAL_BASIC_COMMUNICATION", false);
  // Use a general basic communication if no P in the consumer
  if ((((Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()
         && producer_parallel_desc == consumer_parallel_desc)
        || enable_general_basic_communication)
       && !NdSbpHasPartialParallel(consumer_sbp_parallel))
      && producer_parallel_desc.device_type() == DeviceType::kCUDA
      && consumer_parallel_desc.device_type() == DeviceType::kCUDA) {
    return Cost4GeneralBasicCommunication(producer_sbp_parallel, consumer_sbp_parallel,
                                          logical_blob_desc, producer_parallel_desc,
                                          consumer_parallel_desc)
           + GetTransferCost();
  }
#endif  // WITH_CUDA

  // Not supporting different hierarchy without general basic communication
  if (reduced_in_hierarchy.elem_cnt() != reduced_out_hierarchy.elem_cnt()) {
    return kUnsupportedBoxing;
  }

  double logical_blob_size = TotalByteSize4BlobDesc(logical_blob_desc);

  if (in_dim == 2 && out_dim == 2) {
    // Not supporting different hierarchy
    // TODO: Support it in the future
    if (reduced_in_hierarchy != reduced_out_hierarchy) { return kUnsupportedBoxing; }
    return GetTransferCost()
           + JUST(ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp,
                                                logical_blob_size, reduced_in_hierarchy,
                                                on_same_devices));
  }

  // (in_dim == 2 && out_dim == 1) || (in_dim == 1 && out_dim == 2)
  if (in_dim == 2 && out_dim == 1) {
    return GetTransferCost()
           + JUST(ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp,
                                                logical_blob_size, reduced_in_hierarchy,
                                                on_same_devices));
  }

  if (in_dim == 1 && out_dim == 2) {
    return GetTransferCost()
           + JUST(ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp,
                                                logical_blob_size, reduced_out_hierarchy,
                                                on_same_devices));
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

double GetTransferCost() {
  // Each transfer would have cost.
  // Except for same parallel description and sbp
  static const double kTransferCost = ParseFloatFromEnv("AUTO_PARALLEL_TRANSFER_COST", 1.65e4);
  return kTransferCost;
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
                          const Shape& hierarchy,
                          const HashMap<int32_t, SbpSignatureList>& hierarchy_value2sbp_sig_list,
                          std::vector<NdSbpSignature>* nd_sbp_sig_list) {
  if (depth == dims) {
    nd_sbp_sig_list->push_back(nd_sbp_sig);
  } else {
    for (const auto& sbp_signature :
         hierarchy_value2sbp_sig_list.at(hierarchy.At(depth)).sbp_signature()) {
      SetNdSbpSignature(&nd_sbp_sig, sbp_signature, depth);
      DfsGetNdSbpSignature(nd_sbp_sig, depth + 1, dims, hierarchy, hierarchy_value2sbp_sig_list,
                           nd_sbp_sig_list);
    }
  }
}

namespace {

// give a mesure value for NdSbp for sorting
size_t MesureNdSbp(const NdSbp& nd_sbp) {
  // start from 1, B + P + max split axis (8)
  constexpr size_t kMaxSplitAxis = 8;
  constexpr size_t kCarryDigit = kMaxSplitAxis + 3;
  size_t value = 0;
  for (int i = 0; i < nd_sbp.sbp_parallel_size(); ++i) {
    size_t cur_dim_value = 0;
    const auto& sbp = nd_sbp.sbp_parallel(i);
    if (sbp.has_broadcast_parallel()) {
      cur_dim_value = 1;
    } else if (sbp.has_partial_sum_parallel()) {
      cur_dim_value = 2;
    } else if (sbp.has_split_parallel()) {
      CHECK_LT(sbp.split_parallel().axis(), kMaxSplitAxis);
      // from 3 to 10
      cur_dim_value = 3 + sbp.split_parallel().axis();
    } else {
      UNIMPLEMENTED();
    }
    value = value * kCarryDigit + cur_dim_value;
  }
  return value;
}

size_t MesureNdSbpSignature(const NdSbpSignature& nd_sbp_sig, const std::vector<std::string>& bns) {
  // big enough for 2d-sbp signatrue set
  // if want to extend to 3d-sbp, consider increase to 170
  constexpr size_t kCarryDigit = 97;
  size_t value = 0;
  for (size_t i = 0; i < bns.size(); ++i) {
    auto nd_sbp_it = nd_sbp_sig.bn_in_op2nd_sbp().find(bns[i]);
    CHECK(nd_sbp_it != nd_sbp_sig.bn_in_op2nd_sbp().end())
        << "can't find bn (" << bns[i] << ") in " << PbMessage2TxtString(nd_sbp_sig);
    size_t cur_arg_value = MesureNdSbp(nd_sbp_it->second);
    CHECK_LE(value + cur_arg_value / kCarryDigit, std::numeric_limits<size_t>::max() / kCarryDigit);
    value = value * kCarryDigit + cur_arg_value;
  }
  return value;
}

}  // namespace

void DeduplicateNdSbpSignatureList(std::vector<NdSbpSignature>* nd_sbp_sig_list,
                                   const std::vector<std::string>& bns) {
  if (bns.size() > 8) { return; }
  std::map<size_t, NdSbpSignature> value2nd_sbp_sig;
  for (auto& nd_sbp_sig : *nd_sbp_sig_list) {
    size_t order_value = MesureNdSbpSignature(nd_sbp_sig, bns);
    if (value2nd_sbp_sig.find(order_value) == value2nd_sbp_sig.end()) {
      value2nd_sbp_sig.emplace(order_value, std::move(nd_sbp_sig));
    }
  }
  nd_sbp_sig_list->clear();
  for (auto& nd_sbp_pair : value2nd_sbp_sig) {
    nd_sbp_sig_list->emplace_back(std::move(nd_sbp_pair.second));
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
        if (logical_shape.At(axis) < parallel_hierarchy.At(dim_sbp)) { return kUnsupportedBoxing; }
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
  // In 90% of the transfer, we would have the same parallel description for producer and consumer
  // We need to speed it up and give an approximation of the cost
  if (producer_parallel_desc.EqualsIgnoringHierarchy(consumer_parallel_desc)) {
    // [2, 2]: (S0, S1) -> [2, 2]: (S0, S1)
    if (*producer_parallel_desc.hierarchy() == *consumer_parallel_desc.hierarchy()
        && producer_sbp_parallel == consumer_sbp_parallel) {
      return 0.0;
    }
    // Reduce before cost computation
    Shape reduced_in_hierarchy;
    NdSbp reduced_in_nd_sbp;
    Shape reduced_out_hierarchy;
    NdSbp reduced_out_nd_sbp;
    InOutParallelDimReduce(*producer_parallel_desc.hierarchy(), *consumer_parallel_desc.hierarchy(),
                           producer_sbp_parallel, consumer_sbp_parallel, &reduced_in_hierarchy,
                           &reduced_out_hierarchy, &reduced_in_nd_sbp, &reduced_out_nd_sbp,
                           logical_blob_desc.shape());

    // [2, 2]: (B, B) -> [4]: B
    if (reduced_in_hierarchy == reduced_out_hierarchy && reduced_in_nd_sbp == reduced_out_nd_sbp) {
      return 1.0;
    }
  }
  if (requires_same_sbp) { return kUnsupportedBoxing; }
#ifdef WITH_CUDA
  static const bool enable_general_basic_communication =
      ParseBooleanFromEnv("ONEFLOW_BOXING_ENABLE_GENERAL_BASIC_COMMUNICATION", false);
  // Use a general basic communication if no P in the consumer
  if ((((Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()
         && producer_parallel_desc == consumer_parallel_desc)
        || enable_general_basic_communication)
       && !NdSbpHasPartialParallel(consumer_sbp_parallel))
      && producer_parallel_desc.device_type() == DeviceType::kCUDA
      && consumer_parallel_desc.device_type() == DeviceType::kCUDA) {
    return Cost4GeneralBasicCommunication(producer_sbp_parallel, consumer_sbp_parallel,
                                          logical_blob_desc, producer_parallel_desc,
                                          consumer_parallel_desc)
           + GetTransferCost();
  }
#endif  // WITH_CUDA

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
                                                       requires_same_sbp));
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

// Decide the priority to infer sbp
double ComputeSbpInferPriority(const NdSbp& producer_nd_sbp, const NdSbp& consumer_nd_sbp,
                               const ParallelDesc& producer_parallel_desc,
                               const ParallelDesc& consumer_parallel_desc, bool requires_same_sbp,
                               const Shape& logical_shape) {
  if (producer_nd_sbp == consumer_nd_sbp && producer_parallel_desc == consumer_parallel_desc) {
    // Highest priority: this blob have the same placement and sbp on both the producer and
    // consumer
    return 0.0;
  }
  // Reduce before cost computation
  Shape reduced_in_hierarchy;
  NdSbp reduced_in_nd_sbp;
  Shape reduced_out_hierarchy;
  NdSbp reduced_out_nd_sbp;
  InOutParallelDimReduce(*producer_parallel_desc.hierarchy(), *consumer_parallel_desc.hierarchy(),
                         producer_nd_sbp, consumer_nd_sbp, &reduced_in_hierarchy,
                         &reduced_out_hierarchy, &reduced_in_nd_sbp, &reduced_out_nd_sbp,
                         logical_shape);

  if (requires_same_sbp) {
    // This blob does not support boxing
    if (reduced_in_nd_sbp == reduced_out_nd_sbp && reduced_in_hierarchy == reduced_out_hierarchy
        && producer_parallel_desc.EqualsIgnoringHierarchy(consumer_parallel_desc)) {
      // Normal priority: No transfer occurs but we have different sbp
      // For example: [1]:S0 -> [1]:B
      // [1, 2]:(P, S0) -> [1, 2]:(S0, S0)
      return 1.0;
    } else {
      // Penalty: this blob have different placements and sbps but it does not support boxing
      return 2.0;
    }
  } else {
    // This blob supports boxing
    if (producer_nd_sbp.sbp_parallel_size() == consumer_nd_sbp.sbp_parallel_size()) {
      if (producer_nd_sbp == consumer_nd_sbp) {
        // Highest priority: this blob have the same sbp on both the producer and consumer
        // Not just [0-3] -> [4-7], but also cpu:[0] -> cuda:[0-3]
        return 0.0;
      }
    } else {
      if (reduced_in_nd_sbp == reduced_out_nd_sbp) {
        // Highest priority: this blob have the same sbp on both the producer and consumer
        // [2, 2]: (S0, S0) -> [2]: S0
        // (learning rate) [1]: B -> [2, 2]: (B, B)
        return 0.0;
      }
    }
    // Normal priority: transfer might occurs
    // Or might not: [1, 2]: (P, S0) -> [1, 2]: (B, S0)
    // No transfer but not highest priority
    return 1.0;
  }
}

// The transfer ratio for general basic communication
// Cost = ratio * data amount
// When we get the this function, either producer_sbp_parallel != consumer_sbp_parallel
// or producer_parallel_desc != consumer_parallel_desc
double Cost4GeneralBasicCommunication(const NdSbp& producer_sbp_parallel,
                                      const NdSbp& consumer_sbp_parallel,
                                      const BlobDesc& logical_blob_desc,
                                      const ParallelDesc& producer_parallel_desc,
                                      const ParallelDesc& consumer_parallel_desc) {
  // The upper bound of the amount of the transferred data
  int32_t producer_partial_ratio =
      PartialRatio4Producer(producer_sbp_parallel, producer_parallel_desc);
  int32_t consumer_broadcast_ratio =
      BroadcastRatio4Consumer(consumer_sbp_parallel, consumer_parallel_desc);
  // More intersection on the same devices
  bool on_same_devices = producer_parallel_desc.EqualsIgnoringHierarchy(consumer_parallel_desc);
  // approximate intersection ratio
  double intersection_ratio = 1.0;
  // (?, P, ?)->(Si, Sj)->(?, B, ?), two-step transfer
  if (producer_partial_ratio > 1 && consumer_broadcast_ratio > 1) {
    if (on_same_devices) {
      // Pure P in the producer or B in the consumer
      // (P, P, P) -> ? or ? -> (B, B)
      if (producer_partial_ratio == producer_parallel_desc.parallel_num()
          || consumer_broadcast_ratio == consumer_parallel_desc.parallel_num()) {
        // There some cases which is not applicable to this ratio
        // We just take the one with the largest possibility
        // For example: (P, S0) -> (B, B) for 1-D blob with machine hierarchy [n, m]
        // The path should be (P, S0) -> (S0, S0) -> (B, B)
        // true intersection ratio = 1/m + 1
        intersection_ratio = 2.0;
      } else {
        // sbp_consumer = (B, Si) or (Si, B)
        for (int32_t sbp_id = 0; sbp_id < std::min(producer_sbp_parallel.sbp_parallel_size(),
                                                   consumer_sbp_parallel.sbp_parallel_size());
             sbp_id++) {
          if (consumer_sbp_parallel.sbp_parallel(sbp_id).has_split_parallel()) {
            const auto& producer_sbp4sbp_id = producer_sbp_parallel.sbp_parallel(sbp_id);
            // (B, P) or (Si, P) -> (Si, B)
            // (P, B) or (P, Si) -> (B, Si)
            if (producer_sbp4sbp_id.has_broadcast_parallel()
                || producer_sbp4sbp_id == consumer_sbp_parallel.sbp_parallel(sbp_id)) {
              intersection_ratio = 2.0;
              break;
            }
          }
        }
        // Judge whether the intersection ratio is given a value (2.0)
        if (intersection_ratio == 1.0) {
          // The true intersection ratio range from 0 to 2,
          // we just take a middle point of the range as the approximation
          // For example: (P, S0) -> (S0, B), Path: (P, S0) -> (S1, S0) -> (S0, B)
          // true intersection ratio = 1 + 1/m
          // For example: (P, S0) -> (S1, B), Path: (P, S0) -> (S1, S0) -> (S1, B)
          // true intersection ratio = 1 + 1
          // For example: (P, S0) -> (B, S0), with a 1D blob
          // true intersection ratio = (n+p-1)/nm + (n+p-1)/nm
          // For example: (S0, P) -> (B, S0), Path: (S0, P) -> (S0, S1) -> (B, S0)
          // true intersection ratio = 1 + 1/n

          // We use the approximation 1 + (1/n + 1/m)/2
          intersection_ratio = 1.0 + 0.5 / producer_parallel_desc.hierarchy()->At(0)
                               + 0.5 / producer_parallel_desc.hierarchy()->At(1);
        }
      }
    }
    // Otherwise, on different devices
    // intersection_ratio = 1.0;
  } else {
    // No P in the producer or no B in the consumer, one-step transfer
    if (on_same_devices) {
      // We use simulation for nD sbp with n=1,2,3,...
      TensorSliceView in_second_slice =
          GetTensorSliceView4ParallelId(*producer_parallel_desc.hierarchy(), producer_sbp_parallel,
                                        logical_blob_desc.shape(), /*parallel_id=*/1);
      TensorSliceView out_second_slice =
          GetTensorSliceView4ParallelId(*consumer_parallel_desc.hierarchy(), consumer_sbp_parallel,
                                        logical_blob_desc.shape(), /*parallel_id=*/1);
      const TensorSliceView& intersection = in_second_slice.Intersect(out_second_slice);
      // The intersection ratio is design for two steps.
      // However, we only have one step here, we would increase the ratio by 1.0
      // to eliminate the unused step
      intersection_ratio += std::min(
          1.0, (double)(intersection.shape().elem_cnt() * producer_parallel_desc.parallel_num())
                   / logical_blob_desc.shape().elem_cnt());
    }
    // Otherwise, on different devices
    // intersection_ratio = 1.0;
  }
  // Subtract the intersection part
  return (producer_partial_ratio + consumer_broadcast_ratio - intersection_ratio)
         * TotalByteSize4BlobDesc(logical_blob_desc);
}

}  // namespace oneflow
