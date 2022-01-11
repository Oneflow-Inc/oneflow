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
#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_impl.h"
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"

namespace oneflow {

namespace {

static const double kUnsupportedBoxing = GetMaxVal<float>();

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

Maybe<double> ComputCopyCostBetweenTwoSbpParallel(const cfg::SbpParallel& producer_sbp_parallel,
                                                  const cfg::SbpParallel& consumer_sbp_parallel,
                                                  const BlobDesc& logical_blob_desc,
                                                  const ParallelDesc& producer_parallel_desc,
                                                  const ParallelDesc& consumer_parallel_desc) {
  if (!(CheckSbpParallel(producer_sbp_parallel) && CheckSbpParallel(consumer_sbp_parallel))) {
    return Error::RuntimeError() << "Illegal sbp parallel has been found.";
  }

  if (producer_parallel_desc == consumer_parallel_desc) {
    // Same sbp, no cost: S->S, B->B, P->P
    if (producer_sbp_parallel == consumer_sbp_parallel) { return 0.0; }
    // B->B, B->S, B->P
    if (producer_sbp_parallel.has_broadcast_parallel()) { return 0.0; }

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
    return logical_blob_size * (producer_parallel_desc.parallel_num() - 1);
  } else {
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

}  // namespace

double GetValidMaxCopyCost() {
  // We suppose that valid copy cost range is [0, FloatMax*0.8]
  static const double kValidMaxCopyCost = kUnsupportedBoxing * 0.8;
  return kValidMaxCopyCost;
}

void ResizeNdSbpSignature(cfg::NdSbpSignature& nd_sbp_sig, int32_t size) {
  for (auto& pair : *nd_sbp_sig.mutable_bn_in_op2nd_sbp()) {
    if (pair.second.sbp_parallel_size() > size) { pair.second.clear_sbp_parallel(); }
    while (pair.second.sbp_parallel_size() < size) { pair.second.add_sbp_parallel(); }
  }
}

void SetNdSbpSignature(cfg::NdSbpSignature* nd_sbp_signature,
                       const cfg::SbpSignature& sbp_signature, int32_t sbp_axis) {
  for (const auto& pair : sbp_signature.bn_in_op2sbp_parallel()) {
    *((*nd_sbp_signature->mutable_bn_in_op2nd_sbp())[pair.first].mutable_sbp_parallel(sbp_axis)) =
        pair.second;
  }
}

void DfsGetNdSbpSignature(cfg::NdSbpSignature& nd_sbp_sig, int32_t depth, int32_t dims,
                          const cfg::SbpSignatureList& sbp_sig_list,
                          std::vector<cfg::NdSbpSignature>* nd_sbp_sig_list) {
  if (depth == dims) {
    nd_sbp_sig_list->push_back(nd_sbp_sig);
  } else {
    for (const auto& sbp_signature : sbp_sig_list.sbp_signature()) {
      SetNdSbpSignature(&nd_sbp_sig, sbp_signature, depth);
      DfsGetNdSbpSignature(nd_sbp_sig, depth + 1, dims, sbp_sig_list, nd_sbp_sig_list);
    }
  }
}

Maybe<double> ComputeEagerCopyCostBetweenNdSbp(const cfg::NdSbp& producer_sbp_parallel,
                                               const cfg::NdSbp& consumer_sbp_parallel,
                                               const BlobDesc& logical_blob_desc,
                                               const ParallelDesc& producer_parallel_desc,
                                               const ParallelDesc& consumer_parallel_desc,
                                               bool is_same_sbp) {
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
  cfg::NdSbp reduced_in_nd_sbp;
  cfg::NdSbp reduced_out_nd_sbp;
  InOutParallelDimReduce(producer_parallel_desc, consumer_parallel_desc, producer_sbp_parallel,
                         consumer_sbp_parallel, &reduced_in_parallel_desc,
                         &reduced_out_parallel_desc, &reduced_in_nd_sbp, &reduced_out_nd_sbp);

  const auto& in_hierarchy = reduced_in_parallel_desc.hierarchy();
  const auto& out_hierarchy = reduced_out_parallel_desc.hierarchy();

  bool same_nd_sbp = reduced_in_nd_sbp == reduced_out_nd_sbp;
  // Same sbp is always supported.
  if (same_nd_sbp && reduced_in_parallel_desc == reduced_out_parallel_desc) { return 0.0; }

  // Will directly modify output blob of source op. Requiring data having same sbp_parallel.
  if (is_same_sbp
      && !(reduced_in_parallel_desc.EqualsIgnoringDeviceType(reduced_out_parallel_desc)
           && same_nd_sbp)) {
    return kUnsupportedBoxing;
  }

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
    // nd to nd
    for (int32_t i = 0; i < reduced_in_parallel_desc.hierarchy()->NumAxes(); ++i) {
      const auto& in_sbp = reduced_in_nd_sbp.sbp_parallel(i);
      const auto& out_sbp = reduced_out_nd_sbp.sbp_parallel(i);
      total_cost += JUST(ComputCopyCostBetweenTwoSbpParallel(
          in_sbp, out_sbp, logical_blob_desc, reduced_in_parallel_desc, reduced_out_parallel_desc));
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

Maybe<double> ComputeLazyCopyCostBetweenNdSbp(const cfg::NdSbp& producer_sbp_parallel,
                                              const cfg::NdSbp& consumer_sbp_parallel,
                                              const BlobDesc& logical_blob_desc,
                                              const ParallelDesc& producer_parallel_desc,
                                              const ParallelDesc& consumer_parallel_desc,
                                              bool is_same_sbp) {
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

  bool same_nd_sbp = reduced_in_nd_sbp == reduced_out_nd_sbp;
  if (same_nd_sbp && in_hierarchy == out_hierarchy) { return 0.0; }

  // Will directly modify output blob of source op. Requiring data having same sbp_parallel.
  if (is_same_sbp
      && !(reduced_in_parallel_desc.EqualsIgnoringDeviceType(reduced_out_parallel_desc)
           && same_nd_sbp)) {
    return kUnsupportedBoxing;
  }

  double total_cost = 0.0;
  if (reduced_in_parallel_desc == reduced_out_parallel_desc) {
    // nd to nd
    for (int32_t i = 0; i < reduced_in_parallel_desc.hierarchy()->NumAxes(); ++i) {
      const auto& in_sbp = reduced_in_nd_sbp.sbp_parallel(i);
      const auto& out_sbp = reduced_out_nd_sbp.sbp_parallel(i);
      total_cost += JUST(ComputCopyCostBetweenTwoSbpParallel(
          in_sbp, out_sbp, logical_blob_desc, reduced_in_parallel_desc, reduced_out_parallel_desc));
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

}  // namespace oneflow
