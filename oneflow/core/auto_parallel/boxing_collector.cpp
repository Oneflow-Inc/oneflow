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

#include "oneflow/core/auto_parallel/boxing_collector.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/lazy_mode.h"

namespace oneflow {

namespace {
void DfsSetNdSbp(std::vector<::oneflow::SbpParallel>& id2SbpParallel, int32_t depth,
                 int32_t max_depth, NdSbp& nd_sbp, std::vector<NdSbp>& nd_sbp_lists_,
                 std::unordered_map<::oneflow::NdSbp, int32_t>& NdSbpUniverse_) {
  if (depth == max_depth) {
    NdSbpUniverse_[nd_sbp] = nd_sbp_lists_.size();
    nd_sbp_lists_.push_back(nd_sbp);
  } else {
    for (int32_t i = 0; i < id2SbpParallel.size(); i++) {
      *nd_sbp.mutable_sbp_parallel(depth) = id2SbpParallel[i];
      DfsSetNdSbp(id2SbpParallel, depth + 1, max_depth, nd_sbp, nd_sbp_lists_, NdSbpUniverse_);
    }
  }
}
}  // namespace

// A constructor with init, designed for uncustomized boxing collector
BoxingCollector::BoxingCollector(int32_t max_axis) { CHECK_JUST(Init(max_axis)); }

// Construct a boxing collector with given maximum number of axis
Maybe<void> BoxingCollector::Init(int32_t max_axis) {
  // Set up at least two split for op graph.
  // For a negative example: Resnet50 only have B, P, S(0)
  CollectUniverse(max_axis);
  GenerateNdSbpList();
  JUST(GenerateCombination(2));
  return Maybe<void>::Ok();
}

// Collect Sbp Parallel
void BoxingCollector::CollectUniverse(const SbpParallel& sbp) {
  if (SbpParallelUniverse_.find(sbp) == SbpParallelUniverse_.end()) {
    int32_t curr_size = SbpParallelUniverse_.size();
    SbpParallelUniverse_[sbp] = curr_size;
    id2SbpParallel_.push_back(sbp);
  }
}

// Set default Sbp list
void BoxingCollector::CollectUniverse(int32_t max_axis) {
  SbpParallel sbp;
  sbp.mutable_broadcast_parallel();
  CollectUniverse(sbp);
  for (int32_t axis = 0; axis < max_axis; axis++) {
    sbp.mutable_split_parallel()->set_axis(axis);
    CollectUniverse(sbp);
  }
  sbp.mutable_partial_sum_parallel();
  CollectUniverse(sbp);
}

// Generate nd sbp list
void BoxingCollector::GenerateNdSbpList() {
  // 1D sbp does not support S->P. But it seems that we do not need to deal with it for now.
  // And we do not have 3D sbp or higher dimension.
  int32_t hierarchy_num = 2;

  // Generate possible nd_sbp lists
  NdSbp nd_sbp;
  for (int32_t dim_sbp = 0; dim_sbp < hierarchy_num; dim_sbp++) { nd_sbp.add_sbp_parallel(); }
  DfsSetNdSbp(id2SbpParallel_, 0, hierarchy_num, nd_sbp, nd_sbp_lists_, NdSbpUniverse_);
}

// Generate the transfer rule for different combinations and hierarchies
Maybe<void> BoxingCollector::GenerateCombination(int32_t max_middle_node_num) {
  // other parameters
  // NOTE: The performance of this function are all the same with different hierarchy
  Shape hierarchy44({4, 4});
  std::shared_ptr<Shape> in_hierarchy = std::make_shared<Shape>(hierarchy44);
  auto in_parallel_desc = JUST(ParallelDesc::New("cpu", {"0:0-15"}, in_hierarchy));
  BlobDesc blob_desc({16, 16, 16, 16}, DataType::kInt8, /*is_dynamic=*/false);
  // Store the origin transfer cost information
  int32_t n = nd_sbp_lists_.size();
  minimum_copy_cost_.resize(n);
  middle_nodes_.resize(n);
  for (int32_t i = 0; i < n; i++) {
    minimum_copy_cost_[i].resize(n);
    middle_nodes_[i].resize(n);
    for (int32_t j = 0; j < n; j++) {
      // Get copy cost in lazy mode
      LazyMode::Guard enable_lazy_mode(true);
      minimum_copy_cost_[i][j] = JUST(ComputeLazyCopyCostBetweenNdSbp(
          nd_sbp_lists_[i], nd_sbp_lists_[j], blob_desc, *in_parallel_desc, *in_parallel_desc,
          /*is_same_sbp=*/false));
    }
  }

  auto NotMiddleNode = [&](int32_t i, int32_t j, int32_t k, int32_t middle_node_num_ik) -> bool {
    // Not allow i -> i -> j or i -> j -> j.
    if (k == j || k == i) { return true; }
    // We add middle nodes one by one
    // Thus, we allow multiple nodes from i to k but we only accept 1 step from k to j.
    // i -> ? -> k -> j
    if (middle_nodes_[k][j].size() > 0) { return true; }
    // To avoid multiple counting and bugs, the number of middle nodes between i and k
    // must be exactly middle_node_num_ik, which is (middle_node_num - 1)
    if (middle_node_num_ik) {
      if (middle_nodes_[i][k].size() == 0 || middle_nodes_[i][k][0].size() != middle_node_num_ik) {
        return true;
      }
    } else {
      if (middle_nodes_[i][k].size() > 0) { return true; }
    }
    return false;
  };

  for (int32_t middle_node_num = 1; middle_node_num <= max_middle_node_num; middle_node_num++) {
    int32_t middle_node_num_ik = middle_node_num - 1;

    for (int32_t i = 0; i < n; i++) {
      for (int32_t j = 0; j < n; j++) {
        if (minimum_copy_cost_[i][j] < GetValidMaxCopyCost()) { continue; }
        // Compute the smallest transfer cost
        // k is the middle node, i -> k -> j
        for (int32_t k = 0; k < n; k++) {
          if (NotMiddleNode(i, j, k, middle_node_num_ik)) { continue; }
          double curr_copy_cost = minimum_copy_cost_[i][k] + minimum_copy_cost_[k][j];
          if (curr_copy_cost < minimum_copy_cost_[i][j]) {
            minimum_copy_cost_[i][j] = curr_copy_cost;
          }
        }
        // If the minimum copy cost remians infinity, adding one middle node does not make it.
        if (minimum_copy_cost_[i][j] > GetValidMaxCopyCost()) { continue; }
        // Find those middle nodes
        for (int32_t k = 0; k < n; k++) {
          if (NotMiddleNode(i, j, k, middle_node_num_ik)) { continue; }
          // Now we start to judge if the edge have a minimum cost
          // It needs to be "<=" since we have 0 cost.
          // Using "<" would give no middle nodes from (B, B) to any other nd sbp.
          if (minimum_copy_cost_[i][k] + minimum_copy_cost_[k][j]
              <= minimum_copy_cost_[i][j] * 1.0000001) {
            // i -> ? -> k
            if (middle_nodes_[i][k].size() > 0) {
              // We have multiple choices going from i to k
              for (const auto& middle_node_ik : middle_nodes_[i][k]) {
                middle_nodes_[i][j].push_back(middle_node_ik);
                middle_nodes_[i][j][middle_nodes_[i][j].size() - 1].push_back(k);
              }
            } else {
              // We only need one middle node k to reach j from i
              middle_nodes_[i][j].push_back({k});
            }
          }
        }
        CHECK_OR_RETURN(middle_nodes_[i][j].size() > 0)
            << "No middle nodes given from " << NdSbpToString(nd_sbp_lists_[i]) << " to "
            << NdSbpToString(nd_sbp_lists_[j]) << " in boxing collector";
      }
    }
  }

  return Maybe<void>::Ok();
}

// Print the cost and middle nodes
void BoxingCollector::PrintBoxingTables() {
  if (GlobalProcessCtx::Rank() == 0) {
    LOG(INFO) << "===================minimum copy cost==================" << std::endl;
    // other parameters
    // To be noted that the performance of this function are all the same with different hierarchy
    Shape hierarchy44({4, 4});
    std::shared_ptr<Shape> in_hierarchy = std::make_shared<Shape>(hierarchy44);
    double logical_blob_size = 1024.0;
    int32_t n = nd_sbp_lists_.size();
    // Print the origin copy cost table
    LOG(INFO) << "Cost\t";
    for (int32_t j = 0; j < n; j++) { LOG(INFO) << NdSbpToString(nd_sbp_lists_[j]) << "\t"; }
    LOG(INFO) << std::endl;
    for (int32_t i = 0; i < n; i++) {
      LOG(INFO) << NdSbpToString(nd_sbp_lists_[i]) << "\t";
      for (int32_t j = 0; j < n; j++) {
        if (minimum_copy_cost_[i][j] > GetValidMaxCopyCost()) {
          LOG(INFO) << "X\t";
        } else {
          LOG(INFO) << minimum_copy_cost_[i][j] << "\t";
        }
      }
      LOG(INFO) << std::endl;
    }

    LOG(INFO) << std::endl;
    LOG(INFO) << "Original Copy Cost" << std::endl;
    LOG(INFO) << "logical blob size: " << logical_blob_size << std::endl;
    LOG(INFO) << "hierarchy: " << *in_hierarchy << std::endl;

    LOG(INFO) << "============================middle nodes===========================" << std::endl;

    // Print the middle nodes
    LOG(INFO) << "Middle Sbp\t";
    for (int32_t j = 0; j < n; j++) { LOG(INFO) << NdSbpToString(nd_sbp_lists_[j]) << "\t"; }
    LOG(INFO) << std::endl;
    for (int32_t i = 0; i < n; i++) {
      LOG(INFO) << NdSbpToString(nd_sbp_lists_[i]) << "\t";
      for (int32_t j = 0; j < n; j++) {
        if (minimum_copy_cost_[i][j] > GetValidMaxCopyCost()) {
          LOG(INFO) << "X";
        } else if (middle_nodes_[i][j].size() > 0) {
          for (int32_t k = 0; k < middle_nodes_[i][j].size(); k++) {
            LOG(INFO) << NdSbpToString(nd_sbp_lists_[middle_nodes_[i][j][k][0]]);
            for (int32_t l = 1; l < middle_nodes_[i][j][k].size(); l++) {
              LOG(INFO) << "->" << NdSbpToString(nd_sbp_lists_[middle_nodes_[i][j][k][l]]);
            }
            LOG(INFO) << "; ";
          }
        }

        LOG(INFO) << "\t";
      }
      LOG(INFO) << std::endl;
    }

    LOG(INFO) << std::endl;
    LOG(INFO) << "Minimum Copy Cost after second search" << std::endl;
    LOG(INFO) << "logical blob size: " << logical_blob_size << std::endl;
    LOG(INFO) << "hierarchy: " << *in_hierarchy << std::endl;

    LOG(INFO) << "================================================" << std::endl;
  }
}

// Ask if the boxing algorithm accepts the current sbp combination
Maybe<void> BoxingCollector::AskSbpCombination(const NdSbp& sbp_producer, const NdSbp& sbp_consumer,
                                               const BlobDesc& logical_blob_desc,
                                               const ParallelDesc& producer_parallel_desc,
                                               const ParallelDesc& consumer_parallel_desc,
                                               bool is_customized, std::vector<NdSbp>& middle_sbps,
                                               bool compute_cost) {
  middle_sbps.clear();
  // At this moment, we do not support [2, 3] -> [3, 2]
  // TODO: support [2, 3] -> [3, 2]
  // Middle nodes does not support transfer for different machines or devices or hierarchy
  if (producer_parallel_desc != consumer_parallel_desc) {
    CHECK_OR_RETURN(compute_cost
                    || JUST(ComputeLazyCopyCostBetweenNdSbp(
                           sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc,
                           consumer_parallel_desc, false))
                           < GetValidMaxCopyCost())
        << "Boxing does not support " << NdSbpToString(sbp_producer) << " -> "
        << NdSbpToString(sbp_consumer) << " for two different placement ";
    return Maybe<void>::Ok();
  }
  const auto& parallel_hierarchy = producer_parallel_desc.hierarchy();
  // Dealing with 1D sbp
  if (parallel_hierarchy->NumAxes() == 1) {
    CHECK_OR_RETURN(compute_cost
                    || JUST(ComputeLazyCopyCostBetweenNdSbp(
                           sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc,
                           consumer_parallel_desc, false))
                           < GetValidMaxCopyCost())
        << "Boxing does not support " << NdSbpToString(sbp_producer) << " -> "
        << NdSbpToString(sbp_consumer) << " for 1D sbp";
    return Maybe<void>::Ok();
  }
  // Dealing with nD sbp, n>2
  if (parallel_hierarchy->NumAxes() > 2) {
    CHECK_OR_RETURN(compute_cost)
        << "Boxing does not support a hierarchy with dimension greater than 2";
    return Maybe<void>::Ok();
  }
  // Dealing with 2D sbp
  const auto& it_producer = NdSbpUniverse_.find(sbp_producer);
  const auto& it_consumer = NdSbpUniverse_.find(sbp_consumer);
  if (it_producer != NdSbpUniverse_.end() && it_consumer != NdSbpUniverse_.end()) {
    int32_t i = it_producer->second;
    int32_t j = it_consumer->second;
    // Such combination can not be support with limited middle nodes
    if (minimum_copy_cost_[i][j] > GetValidMaxCopyCost()) {
      CHECK_OR_RETURN(compute_cost) << "Boxing does not support " << NdSbpToString(sbp_producer)
                                    << " -> " << NdSbpToString(sbp_consumer) << " for 2D sbp";
      return Maybe<void>::Ok();
    }
    // Current design can deal with such combination. Do not need to insert middle nodes
    if (middle_nodes_[i][j].size() == 0) { return Maybe<void>::Ok(); }
    // Find a list of middle nodes with minimum storage
    int32_t min_k = -1;
    double min_cost = GetValidMaxCopyCost();
    for (int32_t k = 0; k < middle_nodes_[i][j].size(); k++) {
      double curr_cost = 0.0;
      for (int32_t middle_sbp_id : middle_nodes_[i][j][k]) {
        Shape logical_shape = logical_blob_desc.shape();
        // Storage4NdSbp would modify logical_shape2 as well
        curr_cost +=
            Storage4NdSbp(nd_sbp_lists_[middle_sbp_id], logical_shape, *parallel_hierarchy);
        if (curr_cost > GetValidMaxCopyCost()) { break; }
      }
      // store k if renew minimum cost
      if (curr_cost < min_cost) {
        min_k = k;
        min_cost = curr_cost;
      }
    }

    // If we found a list of middle nodes with current boxing collector
    if (min_k >= 0) {
      for (int32_t middle_sbp_id : middle_nodes_[i][j][min_k]) {
        middle_sbps.push_back(nd_sbp_lists_[middle_sbp_id]);
      }
      return Maybe<void>::Ok();
    }
  }

  // // If we can not found a list of middle nodes even after customized boxing collector
  if (is_customized) {
    CHECK_OR_RETURN(compute_cost) << "Boxing does not support " << NdSbpToString(sbp_producer)
                                  << " -> " << NdSbpToString(sbp_consumer)
                                  << " for Shape: " << logical_blob_desc.shape();
    return Maybe<void>::Ok();
  }

  // Customized boxing collector and try the algorithm again
  BoxingCollector customized_boxing_collector;
  customized_boxing_collector.CollectUniverse(logical_blob_desc.shape().NumAxes());
  customized_boxing_collector.GenerateNdSbpList();
  // Filter out unsuitable middle nodes before computing minimum cost.
  JUST(customized_boxing_collector.FilterNdSbpList4LogicalShape(logical_blob_desc,
                                                                *parallel_hierarchy));
  JUST(customized_boxing_collector.GenerateCombination(5));
  JUST(customized_boxing_collector.AskSbpCombination(sbp_producer, sbp_consumer, logical_blob_desc,
                                                     producer_parallel_desc, consumer_parallel_desc,
                                                     false, middle_sbps, compute_cost));
  return Maybe<void>::Ok();
}

// Filter nd sbp from nd_sbp_lists_ with given logical shape
Maybe<void> BoxingCollector::FilterNdSbpList4LogicalShape(const BlobDesc& logical_blob_desc,
                                                          const Shape& parallel_hierarchy) {
  for (int32_t middle_sbp_id = nd_sbp_lists_.size() - 1; middle_sbp_id >= 0; middle_sbp_id--) {
    Shape logical_shape = logical_blob_desc.shape();
    if (JUST(FilterNdSbpByLogicalShape(nd_sbp_lists_[middle_sbp_id], logical_shape,
                                       parallel_hierarchy))) {
      // Change the value before erasing
      // This might be true: nd_sbp_lists_.size() - 1 == middle_sbp_id
      NdSbpUniverse_[nd_sbp_lists_[nd_sbp_lists_.size() - 1]] = middle_sbp_id;
      NdSbpUniverse_.erase(nd_sbp_lists_[middle_sbp_id]);
      nd_sbp_lists_[middle_sbp_id] = nd_sbp_lists_[nd_sbp_lists_.size() - 1];
      nd_sbp_lists_.pop_back();
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
