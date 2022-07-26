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

  // NOTE: A tensor placed on cpu with a consumer operator that accepts cuda inputs would be
  // transfered to cuda later. We might not have correct parallel description at this moment.
  if (producer_parallel_desc == consumer_parallel_desc) {
    // Same sbp, no cost: S->S, B->B, P->P
    if (producer_sbp_parallel == consumer_sbp_parallel) { return 0.0; }
    double logical_blob_size =
        logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
    // S->P for eager. It should be 0 as well.
    // NOTE: Similar to B->P, we just make the other part to be 0. You can consider P as S(i) for an
    // arbitrary i.
    // ? -> P
    if (consumer_sbp_parallel.has_partial_sum_parallel()) {
      return Penalty4PartialInConsumer(logical_blob_size, producer_parallel_desc.parallel_num(),
                                       consumer_parallel_desc.parallel_num());
    }
    // B->S
    if (producer_sbp_parallel.has_broadcast_parallel()) { return 1.0; }

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
    // ? -> P
    if (consumer_sbp_parallel.has_partial_sum_parallel()) {
      overall_cost +=
          Penalty4PartialInConsumer(logical_blob_size, producer_parallel_desc.parallel_num(),
                                    consumer_parallel_desc.parallel_num());
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
  if (!TRY(Singleton<EagerBoxingInterpreterManager>::Get()->GetEagerBoxingInterpreter(
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

  double total_cost = 1.0;
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
      // Add the penalty for P in the consumer
      if (out_sbp.has_partial_sum_parallel() && (in_sbp != out_sbp)) {
        total_cost += Penalty4PartialInConsumer(
            logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type()),
            producer_parallel_desc.parallel_num(), consumer_parallel_desc.parallel_num());
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

void CollaborativeParallelDimReduce(const ParallelDesc& in_parallel_desc,
                                    const ParallelDesc& out_parallel_desc, const NdSbp& in_nd_sbp,
                                    const NdSbp& out_nd_sbp, ParallelDesc* reduced_in_parallel_desc,
                                    ParallelDesc* reduced_out_parallel_desc,
                                    NdSbp* reduced_in_nd_sbp, NdSbp* reduced_out_nd_sbp) {
  const auto& in_hierarchy = in_parallel_desc.hierarchy();
  const auto& out_hierarchy = out_parallel_desc.hierarchy();
  CHECK_EQ(in_hierarchy->NumAxes(), out_hierarchy->NumAxes());

  DimVector reduced_in_hierarchy;
  DimVector reduced_out_hierarchy;
  FOR_RANGE(int64_t, i, 0, in_hierarchy->NumAxes()) {
    if (in_hierarchy->At(i) != 1 || out_hierarchy->At(i) != 1) {
      if (reduced_in_nd_sbp->sbp_parallel().empty()
          || (in_nd_sbp.sbp_parallel(i)
                  != reduced_in_nd_sbp->sbp_parallel(reduced_in_nd_sbp->sbp_parallel_size() - 1)
              || out_nd_sbp.sbp_parallel(i)
                     != reduced_out_nd_sbp->sbp_parallel(reduced_out_nd_sbp->sbp_parallel_size()
                                                         - 1))) {
        reduced_in_hierarchy.emplace_back(in_hierarchy->At(i));
        *reduced_in_nd_sbp->add_sbp_parallel() = in_nd_sbp.sbp_parallel(i);

        reduced_out_hierarchy.emplace_back(out_hierarchy->At(i));
        *reduced_out_nd_sbp->add_sbp_parallel() = out_nd_sbp.sbp_parallel(i);
      } else {
        reduced_in_hierarchy.back() *= in_hierarchy->At(i);
        reduced_out_hierarchy.back() *= out_hierarchy->At(i);
      }
    }
  }
  if (reduced_in_hierarchy.empty()) {
    reduced_in_hierarchy.emplace_back(in_hierarchy->At(0));
    *reduced_in_nd_sbp->add_sbp_parallel() = in_nd_sbp.sbp_parallel(0);

    reduced_out_hierarchy.emplace_back(out_hierarchy->At(0));
    *reduced_out_nd_sbp->add_sbp_parallel() = out_nd_sbp.sbp_parallel(0);
  }

  ParallelConf reduced_in_parallel_conf = in_parallel_desc.parallel_conf();
  Shape(reduced_in_hierarchy).ToProto(reduced_in_parallel_conf.mutable_hierarchy());
  *reduced_in_parallel_desc = ParallelDesc(reduced_in_parallel_conf);

  ParallelConf reduced_out_parallel_conf = out_parallel_desc.parallel_conf();
  Shape(reduced_out_hierarchy).ToProto(reduced_out_parallel_conf.mutable_hierarchy());
  *reduced_out_parallel_desc = ParallelDesc(reduced_out_parallel_conf);
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

void NdSbpDimReduce(const ParallelDesc& parallel_desc, const NdSbp& nd_sbp,
                    ParallelDesc* reduced_parallel_desc, NdSbp* reduced_nd_sbp) {
  const auto& hierarchy = parallel_desc.hierarchy();
  DimVector reduced_hierarchy;
  FOR_RANGE(int64_t, i, 0, hierarchy->NumAxes()) {
    if (hierarchy->At(i) != 1) {
      if (reduced_nd_sbp->sbp_parallel().empty()
          || (nd_sbp.sbp_parallel(i)
              != reduced_nd_sbp->sbp_parallel(reduced_nd_sbp->sbp_parallel_size() - 1))) {
        reduced_hierarchy.emplace_back(hierarchy->At(i));
        *reduced_nd_sbp->add_sbp_parallel() = nd_sbp.sbp_parallel(i);
      } else {
        reduced_hierarchy.back() *= hierarchy->At(i);
      }
    }
  }
  // [1, 1, ..., 1]: Any --> [1]: (B)
  if (reduced_hierarchy.empty()) {
    reduced_hierarchy.emplace_back(hierarchy->At(0));
    reduced_nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
  }
  ParallelConf reduced_parallel_conf = parallel_desc.parallel_conf();
  Shape(reduced_hierarchy).ToProto(reduced_parallel_conf.mutable_hierarchy());
  *reduced_parallel_desc = ParallelDesc(reduced_parallel_conf);
}

void InOutParallelDimReduce(const ParallelDesc& in_parallel_desc,
                            const ParallelDesc& out_parallel_desc, const NdSbp& in_nd_sbp,
                            const NdSbp& out_nd_sbp, ParallelDesc* reduced_in_parallel_desc,
                            ParallelDesc* reduced_out_parallel_desc, NdSbp* reduced_in_nd_sbp,
                            NdSbp* reduced_out_nd_sbp) {
  const int64_t in_hierarchy_axes = in_parallel_desc.hierarchy()->NumAxes();
  const int64_t out_hierarchy_axes = out_parallel_desc.hierarchy()->NumAxes();
  if (in_hierarchy_axes == 1 && out_hierarchy_axes == 1) {
    *reduced_in_parallel_desc = in_parallel_desc;
    *reduced_out_parallel_desc = out_parallel_desc;
    *reduced_in_nd_sbp = in_nd_sbp;
    *reduced_out_nd_sbp = out_nd_sbp;
  } else if (in_hierarchy_axes != out_hierarchy_axes) {
    NdSbpDimReduce(in_parallel_desc, in_nd_sbp, reduced_in_parallel_desc, reduced_in_nd_sbp);
    NdSbpDimReduce(out_parallel_desc, out_nd_sbp, reduced_out_parallel_desc, reduced_out_nd_sbp);
  } else {
    CollaborativeParallelDimReduce(in_parallel_desc, out_parallel_desc, in_nd_sbp, out_nd_sbp,
                                   reduced_in_parallel_desc, reduced_out_parallel_desc,
                                   reduced_in_nd_sbp, reduced_out_nd_sbp);
  }
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
    return GetTransferCost()
           + JUST(ComputCopyCostBetweenTwoSbpParallel(
               reduced_in_nd_sbp.sbp_parallel(0), reduced_out_nd_sbp.sbp_parallel(0),
               logical_blob_desc, reduced_in_parallel_desc, reduced_out_parallel_desc));
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
  if (in_hierarchy->elem_cnt() != out_hierarchy->elem_cnt()) { return kUnsupportedBoxing; }

  bool on_same_devices =
      reduced_in_parallel_desc.EqualsIgnoringHierarchy(reduced_out_parallel_desc);
  double logical_blob_size =
      logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());

  if (in_dim == 2 && out_dim == 2) {
    // Not supporting different hierarchy
    // TODO: Support it in the future
    if (*in_hierarchy != *out_hierarchy) { return kUnsupportedBoxing; }
    return GetTransferCost()
           + JUST(ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp,
                                                logical_blob_size, in_hierarchy, on_same_devices));
  }

  // (in_dim == 2 && out_dim == 1) || (in_dim == 1 && out_dim == 2)
  if (in_dim == 2 && out_dim == 1) {
    return GetTransferCost()
           + JUST(ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp,
                                                logical_blob_size, in_hierarchy, on_same_devices));
  }

  if (in_dim == 1 && out_dim == 2) {
    return GetTransferCost()
           + JUST(ComputCopyCostBetweenTwoNdSbp(reduced_in_nd_sbp, reduced_out_nd_sbp,
                                                logical_blob_size, out_hierarchy, on_same_devices));
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
  static const double kTransferCost = ParseFloatFromEnv("AUTO_PARALLEL_TRANSFER_COST", 1.65e8);
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
  // Reduce before cost computation
  ParallelDesc reduced_in_parallel_desc = producer_parallel_desc;
  NdSbp reduced_in_nd_sbp;
  NdSbpDimReduce(producer_parallel_desc, producer_sbp_parallel, &reduced_in_parallel_desc,
                 &reduced_in_nd_sbp);

  ParallelDesc reduced_out_parallel_desc = consumer_parallel_desc;
  NdSbp reduced_out_nd_sbp;
  NdSbpDimReduce(consumer_parallel_desc, consumer_sbp_parallel, &reduced_out_parallel_desc,
                 &reduced_out_nd_sbp);
  // In 90% of the transfer, we would have the same parallel description for producer and consumer
  // We need to speed it up and give an approximation of the cost
  if (reduced_in_parallel_desc == reduced_out_parallel_desc
      && reduced_in_nd_sbp == reduced_out_nd_sbp) {
    return 0.0;
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
                               const ParallelDesc& consumer_parallel_desc, bool requires_same_sbp) {
  if (producer_nd_sbp == consumer_nd_sbp && producer_parallel_desc == consumer_parallel_desc) {
    // Highest priority: this blob have the same placement and sbp on both the producer and
    // consumer
    return 0.0;
  }
  // Dim reduction for producer
  ParallelDesc reduced_in_parallel_desc = producer_parallel_desc;
  NdSbp reduced_in_nd_sbp;
  NdSbpDimReduce(producer_parallel_desc, producer_nd_sbp, &reduced_in_parallel_desc,
                 &reduced_in_nd_sbp);

  // Dim reduction for consumer
  ParallelDesc reduced_out_parallel_desc = consumer_parallel_desc;
  NdSbp reduced_out_nd_sbp;
  NdSbpDimReduce(consumer_parallel_desc, consumer_nd_sbp, &reduced_out_parallel_desc,
                 &reduced_out_nd_sbp);

  if (requires_same_sbp) {
    // This blob does not support boxing
    if (reduced_in_nd_sbp == reduced_out_nd_sbp
        && reduced_in_parallel_desc == reduced_out_parallel_desc) {
      // Normal priority: No transfer occurs but we have different sbp
      // For example: [1]:S0 -> [1]:B
      // [1, 2]:(P, S0) -> [1, 2]:(S0, S0)
      return 1.0;
    } else {
      // Penality: this blob have different placements and sbps but it does not support boxing
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
         * logical_blob_desc.shape().elem_cnt() * GetSizeOfDataType(logical_blob_desc.data_type());
}

}  // namespace oneflow
