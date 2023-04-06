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
#include <string>
#include "oneflow/core/auto_parallel/algorithm_util.h"
#include "oneflow/core/auto_parallel/boxing_collector.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/lazy_mode.h"

namespace oneflow {

namespace {

static bool disable_middle_node = false;

void DfsSetNdSbp(const std::vector<SbpParallel>& id2sbp_parallel, int32_t depth, int32_t max_depth,
                 NdSbp& nd_sbp, std::vector<NdSbp>& nd_sbp_lists,
                 std::unordered_map<NdSbp, int32_t>& nd_sbp_universe) {
  if (depth == max_depth) {
    nd_sbp_universe[nd_sbp] = nd_sbp_lists.size();
    nd_sbp_lists.push_back(nd_sbp);
  } else {
    for (const auto& sbp_parallel : id2sbp_parallel) {
      *nd_sbp.mutable_sbp_parallel(depth) = sbp_parallel;
      DfsSetNdSbp(id2sbp_parallel, depth + 1, max_depth, nd_sbp, nd_sbp_lists, nd_sbp_universe);
    }
  }
}

// Let a nd sbp be consistent with the given hierarchy number
Maybe<NdSbp> SetNdSbpDim(const NdSbp& nd_sbp, int32_t hierarchy_num) {
  // Do not need to change
  if (nd_sbp.sbp_parallel_size() == hierarchy_num) { return nd_sbp; }
  // (S0, S0) -> S0
  if (hierarchy_num == 1) {
    CHECK_OR_RETURN(Is1dSbp(nd_sbp))
        << NdSbpToString(nd_sbp) << " can not be converted to a 1d sbp!";
    NdSbp new_sbp;
    new_sbp.add_sbp_parallel();
    *new_sbp.mutable_sbp_parallel(0) = nd_sbp.sbp_parallel(0);
    return new_sbp;
  }
  // S0 -> (S0, S0)
  CHECK_EQ_OR_RETURN(nd_sbp.sbp_parallel_size(), 1) << "Illegal nd sbp transform.";
  NdSbp new_sbp;
  for (int32_t i = 0; i < hierarchy_num; i++) {
    new_sbp.add_sbp_parallel();
    *new_sbp.mutable_sbp_parallel(i) = nd_sbp.sbp_parallel(0);
  }
  return new_sbp;
}

int32_t TotalNumSplit(const NdSbp& nd_sbp, const ParallelDesc& parallel_desc) {
  int32_t total_num_split = 1;
  for (int32_t i = 0; i < nd_sbp.sbp_parallel_size(); i++) {
    if (nd_sbp.sbp_parallel(i).has_split_parallel()) {
      total_num_split *= parallel_desc.hierarchy()->At(i);
    }
  }
  return total_num_split;
}

// Dealing with 1D sbp to 1D sbp
// Specifically, S -> P.
Maybe<void> AskSbpCombinationFor1DSbp(const NdSbp& sbp_producer, const NdSbp& sbp_consumer,
                                      const ParallelDesc& producer_parallel_desc,
                                      const ParallelDesc& consumer_parallel_desc,
                                      std::vector<NdSbp>& middle_sbps, int32_t* diag_node_pos) {
  if (sbp_consumer.sbp_parallel(0).has_partial_sum_parallel()) {
    // Support [4]: P <--> [2, 2]: (P, P)
    // Support {0, 1, 2, 3}: P <--> {2, 0, 6, 7}: (P, P)
    if (producer_parallel_desc.parallel_num() == consumer_parallel_desc.parallel_num()
        && sbp_producer.sbp_parallel(0).has_partial_sum_parallel()) {
      return Maybe<void>::Ok();
    }

    if (!sbp_producer.sbp_parallel(0).has_broadcast_parallel()) {
      // S -> B -> P (Large cost!)
      // TODO: Please implement S -> P directly.
      // We do not support [3]: P <--> [2, 2]: (P, P) as well.

      int32_t hierarchy_size = 0;
      if (producer_parallel_desc.hierarchy()->elem_cnt()
          < consumer_parallel_desc.hierarchy()->elem_cnt()) {
        // The diagonal node uses the parallel description from producer
        // (S, S) -> (B, B) -> P/(P, P) or S -> B -> P/(P, P)
        *diag_node_pos = 1;
        hierarchy_size = producer_parallel_desc.hierarchy()->NumAxes();
      } else {
        // The diagonal node uses the parallel description from consumer
        // S/(S, S) -> B -> P or S/(S, S) -> (B, B) -> (P, P)
        *diag_node_pos = 0;
        hierarchy_size = consumer_parallel_desc.hierarchy()->NumAxes();
      }

      NdSbp broadcast_nd;
      for (int32_t i = 0; i < hierarchy_size; i++) {
        broadcast_nd.add_sbp_parallel();
        broadcast_nd.mutable_sbp_parallel(i)->mutable_broadcast_parallel();
      }
      middle_sbps.emplace_back(broadcast_nd);
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

// A constructor with init, designed for pre-stored boxing collector
BoxingCollector::BoxingCollector(int32_t max_axis) { CHECK_JUST(Init(max_axis)); }

// Construct a boxing collector with given maximum number of axis
Maybe<void> BoxingCollector::Init(int32_t max_axis) {
  // Update environment parameter
  disable_middle_node = ParseBooleanFromEnv("ONEFLOW_BOXING_DISABLE_MIDDLE_NODE_AND_CHECK", false);
  // Not allowed two-step boxing and disable checking for debugging
  if (disable_middle_node) { return Maybe<void>::Ok(); }
  // Set up at least two split for op graph.
  // For a negative example: Resnet50 only have B, P, S(0)
  CollectUniverse(max_axis);
  GenerateNdSbpList(2);
  GenerateMap1d2nd();
  // Get copy cost in lazy mode
  LazyMode::Guard enable_lazy_mode(true);
  JUST(GenerateCombination4SamePlacement(3));
  JUST(GenerateCombination4DiffHierarchy(this, this));
  JUST(GenerateCombination4DiffPlacement(this, this));
  init_type_ = int32_t(enable_general_basic_communication
                       || Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream());
  return Maybe<void>::Ok();
}

// Customized initialization with given blob and parallel description
Maybe<void> BoxingCollector::Init(const BlobDesc& logical_blob_desc,
                                  const ParallelDesc& parallel_desc) {
  CollectUniverse(logical_blob_desc.shape().NumAxes());
  GenerateNdSbpList(parallel_desc.hierarchy()->NumAxes());
  // Filter out unsuitable middle nodes before computing minimum cost.
  JUST(FilterNdSbpList4LogicalShape(logical_blob_desc, *parallel_desc.hierarchy()));
  GenerateMap1d2nd();
  // Get copy cost in lazy mode
  LazyMode::Guard enable_lazy_mode(true);
  JUST(GenerateCombination4SamePlacement(5, logical_blob_desc, parallel_desc));
  init_type_ = int32_t(enable_general_basic_communication
                       || Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream());
  return Maybe<void>::Ok();
}

// Collect Sbp Parallel
void BoxingCollector::CollectUniverse(const SbpParallel& sbp) {
  if (sbp_parallel_universe_.find(sbp) == sbp_parallel_universe_.end()) {
    int32_t curr_size = sbp_parallel_universe_.size();
    sbp_parallel_universe_[sbp] = curr_size;
    id2sbp_parallel_.push_back(sbp);
  }
}

// Find corresponding id for Nd sbp
int32_t BoxingCollector::FindId4NdSbp(const NdSbp& nd_sbp) {
  // Directly search on the nd_sbp_list
  if (nd_sbp.sbp_parallel_size() == hierarchy_num_) {
    const auto& it_nd_sbp = nd_sbp_universe_.find(nd_sbp);
    if (it_nd_sbp != nd_sbp_universe_.end()) {
      return it_nd_sbp->second;
    } else {
      return -1;
    }
  }

  // Find the diagonal node if it could be converted to a 1D sbp
  if (Is1dSbp(nd_sbp)) {
    const auto& it_nd_sbp = sbp_parallel_universe_.find(nd_sbp.sbp_parallel(0));
    if (it_nd_sbp != sbp_parallel_universe_.end()) { return id_1d_2_nd_[it_nd_sbp->second]; }
  }

  // Can not be converted to a 1D sbp or not found in the 1D sbp list
  return -1;
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
void BoxingCollector::GenerateNdSbpList(int32_t hierarchy_num) {
  // 1D sbp does not support S->P. But it seems that we do not need to deal with it for now.
  // And we do not have 3D sbp or higher dimension.
  hierarchy_num_ = hierarchy_num;

  // Generate possible nd_sbp lists
  NdSbp nd_sbp;
  for (int32_t dim_sbp = 0; dim_sbp < hierarchy_num; dim_sbp++) { nd_sbp.add_sbp_parallel(); }
  DfsSetNdSbp(id2sbp_parallel_, 0, hierarchy_num, nd_sbp, nd_sbp_lists_, nd_sbp_universe_);
}

// Generate the map from 1d sbp to 2d sbp
void BoxingCollector::GenerateMap1d2nd() {
  // Number of 1d sbp
  int32_t m = id2sbp_parallel_.size();

  // Generate the id Map from 1d sbp to nd sbp
  NdSbp nd_sbp;
  for (int32_t dim_sbp = 0; dim_sbp < hierarchy_num_; dim_sbp++) { nd_sbp.add_sbp_parallel(); }
  id_1d_2_nd_.clear();
  id_1d_2_nd_.resize(m, -1);
  for (int32_t id_1d = 0; id_1d < m; id_1d++) {
    for (int32_t dim_sbp = 0; dim_sbp < hierarchy_num_; dim_sbp++) {
      *nd_sbp.mutable_sbp_parallel(dim_sbp) = id2sbp_parallel_[id_1d];
    }
    // NOTE: The 2d sbp might be filtered out already.
    const auto& it_ = nd_sbp_universe_.find(nd_sbp);
    if (it_ != nd_sbp_universe_.end()) { id_1d_2_nd_[id_1d] = it_->second; }
  }
}

// Generate the transfer rule for different combinations with the same hierarchy
Maybe<void> BoxingCollector::GenerateCombination4SamePlacement(int32_t max_middle_node_num) {
  // other parameters
  // NOTE: The performance of this function are all the same with different hierarchy
  int32_t world_size = GlobalProcessCtx::WorldSize();
  Shape hierarchy44({4 * world_size, 4 * world_size});
  int32_t virtual_range_size = hierarchy44.elem_cnt();
  std::shared_ptr<Shape> virtual_hierarchy = std::make_shared<Shape>(hierarchy44);
  auto parallel_desc = JUST(ParallelDesc::New(
      "cpu", {"0:0-" + std::to_string(hierarchy44.elem_cnt() - 1)}, virtual_hierarchy));
  BlobDesc blob_desc({virtual_range_size, virtual_range_size, virtual_range_size,
                      virtual_range_size, virtual_range_size, virtual_range_size},
                     DataType::kInt8, /*is_dynamic=*/false);
  JUST(GenerateCombination4SamePlacement(max_middle_node_num, blob_desc, *parallel_desc));
  return Maybe<void>::Ok();
}

// Generate the transfer rule for different combinations with the same hierarchy
Maybe<void> BoxingCollector::GenerateCombination4SamePlacement(int32_t max_middle_node_num,
                                                               const BlobDesc& blob_desc,
                                                               const ParallelDesc& parallel_desc) {
  // Store the origin transfer cost information
  int32_t n = nd_sbp_lists_.size();
  minimum_copy_cost_.clear();
  minimum_copy_cost_.resize(n);
  middle_nodes_.clear();
  middle_nodes_.resize(n);
  for (int32_t i = 0; i < n; i++) {
    minimum_copy_cost_[i].resize(n);
    middle_nodes_[i].resize(n);
    for (int32_t j = 0; j < n; j++) {
      minimum_copy_cost_[i][j] = JUST(ComputeLazyCopyCostBetweenNdSbp(
          nd_sbp_lists_[i], nd_sbp_lists_[j], blob_desc, parallel_desc, parallel_desc,
          /*requires_same_sbp=*/false));
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
        // If the minimum copy cost remains infinity, adding one middle node does not make it.
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

// Generate the transfer rule for different combinations with different hierarchies on the same
// placement
Maybe<void> BoxingCollector::GenerateCombination4DiffHierarchy(
    BoxingCollector* boxing_collector_producer, BoxingCollector* boxing_collector_consumer) {
  // Store the boxing collector pointer

  // Search the path that contains one of the diagonal sbp
  int32_t n = nd_sbp_lists_.size();
  diag_node_diff_hierarchy_.clear();
  diag_node_diff_hierarchy_.resize(n);
  for (int32_t i = 0; i < n; i++) {
    diag_node_diff_hierarchy_[i].resize(n);
    for (int32_t j = 0; j < n; j++) {
      JUST(Generate1Combination4DiffHierarchy(i, j, boxing_collector_producer,
                                              boxing_collector_consumer,
                                              diag_node_diff_hierarchy_[i][j]));
    }
  }

  return Maybe<void>::Ok();
}

// Generate the transfer rule for different combinations with different placements
Maybe<void> BoxingCollector::GenerateCombination4DiffPlacement(
    BoxingCollector* boxing_collector_producer, BoxingCollector* boxing_collector_consumer) {
  // Virtual parallel and blob description
  int32_t world_size = GlobalProcessCtx::WorldSize();
  int32_t virtual_range_size = 4 * world_size * (4 * world_size + 1);
  BlobDesc blob_desc({virtual_range_size, virtual_range_size, virtual_range_size,
                      virtual_range_size, virtual_range_size, virtual_range_size},
                     DataType::kInt8, /*is_dynamic=*/false);
  // Virtual placements before transfer
  Shape in_hierarchy44({4 * world_size + 1, 4 * world_size});
  std::shared_ptr<Shape> in_hierarchy = std::make_shared<Shape>(in_hierarchy44);
  auto in_parallel_desc = JUST(ParallelDesc::New(
      "cpu", {"0:0-" + std::to_string(in_hierarchy44.elem_cnt() - 1)}, in_hierarchy));
  // Virtual placements after transfer
  Shape out_hierarchy44({4 * world_size, 4 * world_size});
  std::shared_ptr<Shape> out_hierarchy = std::make_shared<Shape>(out_hierarchy44);
  auto out_parallel_desc = JUST(ParallelDesc::New(
      "cpu", {"0:0-" + std::to_string(out_hierarchy44.elem_cnt() - 1)}, out_hierarchy));

  JUST(GenerateCombination4DiffPlacement(boxing_collector_producer, boxing_collector_consumer,
                                         blob_desc, *in_parallel_desc, *out_parallel_desc));
  return Maybe<void>::Ok();
}

// The cost for transferring a 1D sbp between different placements
Maybe<void> BoxingCollector::ComputeCostFor1DSbpDiffPlacement(
    const BlobDesc& blob_desc, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc,
    std::vector<std::vector<double>>& cost_4_diff_placement) {
  // Number of 1d sbp
  int32_t m = id2sbp_parallel_.size();
  // Compute the cost while transferring a 1D sbp between different placements
  cost_4_diff_placement.clear();
  cost_4_diff_placement.resize(m);
  for (int32_t id_1d_producer = 0; id_1d_producer < m; id_1d_producer++) {
    cost_4_diff_placement[id_1d_producer].resize(m, GetMaxVal<float>());
    int32_t diag_producer = id_1d_2_nd_[id_1d_producer];
    if (diag_producer < 0) { continue; }

    for (int32_t id_1d_consumer = 0; id_1d_consumer < m; id_1d_consumer++) {
      int32_t diag_consumer = id_1d_2_nd_[id_1d_consumer];
      if (diag_consumer < 0) { continue; }
      cost_4_diff_placement[id_1d_producer][id_1d_consumer] = JUST(ComputeLazyCopyCostBetweenNdSbp(
          nd_sbp_lists_[diag_producer], nd_sbp_lists_[diag_consumer], blob_desc, in_parallel_desc,
          out_parallel_desc, false));
    }
  }
  return Maybe<void>::Ok();
}

// Generate the transfer rule for different combinations with different placements
Maybe<void> BoxingCollector::GenerateCombination4DiffPlacement(
    BoxingCollector* boxing_collector_producer, BoxingCollector* boxing_collector_consumer,
    const BlobDesc& blob_desc, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc) {
  // The cost for transferring a 1D sbp between different placements
  std::vector<std::vector<double>> cost_4_diff_placement;
  // Compute the cost while transferring a 1D sbp between different placements
  JUST(ComputeCostFor1DSbpDiffPlacement(blob_desc, in_parallel_desc, out_parallel_desc,
                                        cost_4_diff_placement));

  // Search the path that contains two of the diagonal sbp
  int32_t n = nd_sbp_lists_.size();
  diag_node_diff_placement_.clear();
  diag_node_diff_placement_.resize(n);
  for (int32_t i = 0; i < n; i++) {
    diag_node_diff_placement_[i].resize(n);
    for (int32_t j = 0; j < n; j++) {
      JUST(Generate1Combination4DiffPlacement(i, j, boxing_collector_producer,
                                              boxing_collector_consumer, cost_4_diff_placement,
                                              diag_node_diff_placement_[i][j]));
    }
  }

  return Maybe<void>::Ok();
}

// Print the cost and middle nodes
void BoxingCollector::PrintBoxingTables() {
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "===================minimum copy cost==================" << std::endl;
    // other parameters
    // To be noted that the performance of this function are all the same with different hierarchy
    Shape hierarchy44({4, 4});
    std::shared_ptr<Shape> in_hierarchy = std::make_shared<Shape>(hierarchy44);
    double logical_blob_size = 1024.0;
    int32_t n = nd_sbp_lists_.size();
    // Print the origin copy cost table
    std::cout << "Cost\t";
    for (int32_t j = 0; j < n; j++) { std::cout << NdSbpToString(nd_sbp_lists_[j]) << "\t"; }
    std::cout << std::endl;
    for (int32_t i = 0; i < n; i++) {
      std::cout << NdSbpToString(nd_sbp_lists_[i]) << "\t";
      for (int32_t j = 0; j < n; j++) {
        if (minimum_copy_cost_[i][j] > GetValidMaxCopyCost()) {
          std::cout << "X\t";
        } else {
          std::cout << minimum_copy_cost_[i][j] << "\t";
        }
      }
      std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Original Copy Cost" << std::endl;
    std::cout << "logical blob size: " << logical_blob_size << std::endl;
    std::cout << "hierarchy: " << *in_hierarchy << std::endl;

    std::cout << "============================middle nodes===========================" << std::endl;

    // Print the middle nodes
    std::cout << "Middle Sbp\t";
    for (int32_t j = 0; j < n; j++) { std::cout << NdSbpToString(nd_sbp_lists_[j]) << "\t"; }
    std::cout << std::endl;
    for (int32_t i = 0; i < n; i++) {
      std::cout << NdSbpToString(nd_sbp_lists_[i]) << "\t";
      for (int32_t j = 0; j < n; j++) {
        if (minimum_copy_cost_[i][j] > GetValidMaxCopyCost()) {
          std::cout << "X";
        } else if (middle_nodes_[i][j].size() > 0) {
          for (int32_t k = 0; k < middle_nodes_[i][j].size(); k++) {
            std::cout << NdSbpToString(nd_sbp_lists_[middle_nodes_[i][j][k][0]]);
            for (int32_t l = 1; l < middle_nodes_[i][j][k].size(); l++) {
              std::cout << "->" << NdSbpToString(nd_sbp_lists_[middle_nodes_[i][j][k][l]]);
            }
            std::cout << "; ";
          }
        }

        std::cout << "\t";
      }
      std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Minimum Copy Cost after second search" << std::endl;
    std::cout << "logical blob size: " << logical_blob_size << std::endl;
    std::cout << "hierarchy: " << *in_hierarchy << std::endl;

    std::cout << "====================middle nodes for different placement===================="
              << std::endl;

    std::cout << "Middle nodes for different placement\t";
    for (int32_t j = 0; j < n; j++) { std::cout << NdSbpToString(nd_sbp_lists_[j]) << "\t"; }
    std::cout << std::endl;
    for (int32_t i = 0; i < n; i++) {
      std::cout << NdSbpToString(nd_sbp_lists_[i]) << "\t";
      for (int32_t j = 0; j < n; j++) {
        if (diag_node_diff_placement_[i][j].size() > 0) {
          for (int32_t k = 0; k < diag_node_diff_placement_[i][j].size(); k++) {
            std::cout << "[" << NdSbpToString(nd_sbp_lists_[diag_node_diff_placement_[i][j][k][0]])
                      << ", " << NdSbpToString(nd_sbp_lists_[diag_node_diff_placement_[i][j][k][1]])
                      << "]; ";
          }
        }
        std::cout << "\t";
      }
      std::cout << std::endl;
    }

    std::cout << "====================middle nodes for different hierarchy===================="
              << std::endl;

    std::cout << "Middle nodes for different hierarchy\t";
    for (int32_t j = 0; j < n; j++) { std::cout << NdSbpToString(nd_sbp_lists_[j]) << "\t"; }
    std::cout << std::endl;
    for (int32_t i = 0; i < n; i++) {
      std::cout << NdSbpToString(nd_sbp_lists_[i]) << "\t";
      for (int32_t j = 0; j < n; j++) {
        if (diag_node_diff_hierarchy_[i][j].size() > 0) {
          for (int32_t k = 0; k < diag_node_diff_hierarchy_[i][j].size(); k++) {
            std::cout << NdSbpToString(nd_sbp_lists_[diag_node_diff_hierarchy_[i][j][k][0]])
                      << "; ";
          }
        }
        std::cout << "\t";
      }
      std::cout << std::endl;
    }

    std::cout << "================================================" << std::endl;
  }
}

// Ask if the boxing algorithm accepts the current sbp combination
Maybe<void> BoxingCollector::AskSbpCombination(const NdSbp& sbp_producer, const NdSbp& sbp_consumer,
                                               const BlobDesc& logical_blob_desc,
                                               const ParallelDesc& producer_parallel_desc,
                                               const ParallelDesc& consumer_parallel_desc,
                                               bool is_customized, std::vector<NdSbp>& middle_sbps,
                                               int32_t* diag_node_pos, bool compute_cost) {
  middle_sbps.clear();
  // Not allowed two-step boxing and disable checking for debugging
  if (disable_middle_node) { return Maybe<void>::Ok(); }
  if (producer_parallel_desc == consumer_parallel_desc && sbp_producer == sbp_consumer) {
    return Maybe<void>::Ok();
  }

  // Dealing with 1D sbp to 1D sbp
  if (Is1dSbp(sbp_producer) && Is1dSbp(sbp_consumer)) {
    JUST(AskSbpCombinationFor1DSbp(sbp_producer, sbp_consumer, producer_parallel_desc,
                                   consumer_parallel_desc, middle_sbps, diag_node_pos));
    // No middle nodes for the other 1d-sbp combinations
    return Maybe<void>::Ok();
  }

#ifdef WITH_CUDA
  // Use a general basic communication if no P in the consumer
  if (((Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()
        && producer_parallel_desc == consumer_parallel_desc)
       || enable_general_basic_communication)
      && (!NdSbpHasPartialParallel(sbp_consumer))
      && producer_parallel_desc.device_type() == DeviceType::kCUDA
      && consumer_parallel_desc.device_type() == DeviceType::kCUDA) {
    if (NdSbpHasPartialParallel(sbp_producer) && NdSbpHasBroadcastParallel(sbp_consumer)) {
      // (?, P, ?)->(Si, Sj)->(?, B, ?), two-step transfer
      // Directly applying general basic communication would have O(n^2) time complexity for P->B
      // Using two-step transfer would reduce it to a linear cost
      JUST(AskSbpCombination4GeneralBasicCommunication(
          sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc,
          consumer_parallel_desc, middle_sbps, diag_node_pos));
    }
    // Otherwise, one-step transfer
    return Maybe<void>::Ok();
  }
#endif  // WITH_CUDA

  if (JUST(ComputeLazyCopyCostBetweenNdSbp(sbp_producer, sbp_consumer, logical_blob_desc,
                                           producer_parallel_desc, consumer_parallel_desc,
                                           /*requires_same_sbp=*/false))
      < GetValidMaxCopyCost()) {
    return Maybe<void>::Ok();
  } else {
    int32_t require_init_type =
        int32_t(enable_general_basic_communication
                || Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream());
    if (init_type_ != require_init_type) {
      // We assemble the boxing table from S(0) to S(5).
      // Those splitting in higher axes are considered in the customized boxing.
      constexpr int32_t kRegularMaxSplitAxes = 6;
      JUST(Init(kRegularMaxSplitAxes));
    }
  }

  // Middle nodes algorithm supports transfer for different machines or devices or hierarchies
  if (producer_parallel_desc != consumer_parallel_desc) {
    JUST(AskSbpCombination4DiffPlacement(sbp_producer, sbp_consumer, logical_blob_desc,
                                         producer_parallel_desc, consumer_parallel_desc,
                                         is_customized, middle_sbps, diag_node_pos, compute_cost));

    return Maybe<void>::Ok();
  }
  // Transfer for the same machines, devices and hierarchy.
  if (sbp_producer == sbp_consumer) { return Maybe<void>::Ok(); }
  const auto& parallel_hierarchy = producer_parallel_desc.hierarchy();

  *diag_node_pos = 0;
  // Dealing with nD sbp, n>2
  if (parallel_hierarchy->NumAxes() > 2) {
    CHECK_OR_RETURN(compute_cost)
        << "Boxing does not support a hierarchy with dimension greater than 2";
    return Maybe<void>::Ok();
  }
  // Ask for sbp combination with the same 2-D hierarchy and placement
  JUST(AskSbpCombination4Same2DPlacement(sbp_producer, sbp_consumer, logical_blob_desc,
                                         producer_parallel_desc, consumer_parallel_desc,
                                         is_customized, middle_sbps, diag_node_pos, compute_cost));

  return Maybe<void>::Ok();
}

// Ask for sbp combination with the same 2-D hierarchy and placement
Maybe<void> BoxingCollector::AskSbpCombination4Same2DPlacement(
    const NdSbp& sbp_producer, const NdSbp& sbp_consumer, const BlobDesc& logical_blob_desc,
    const ParallelDesc& producer_parallel_desc, const ParallelDesc& consumer_parallel_desc,
    bool is_customized, std::vector<NdSbp>& middle_sbps, int32_t* diag_node_pos,
    bool compute_cost) {
  CHECK_OR_RETURN(producer_parallel_desc == consumer_parallel_desc)
      << "Producer and consumer have different placements, Please use AskSbpCombination directly";
  middle_sbps.clear();

  // Find the 2D sbp id
  int32_t i = FindId4NdSbp(sbp_producer);
  int32_t j = FindId4NdSbp(sbp_consumer);
  // Dealing with 2D sbp
  if (i >= 0 && j >= 0) {
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
        curr_cost += Storage4NdSbp(nd_sbp_lists_[middle_sbp_id], logical_shape,
                                   *producer_parallel_desc.hierarchy());
        if (curr_cost > GetValidMaxCopyCost()) { break; }
      }
      // store k if renew minimum cost
      if (curr_cost < min_cost) {
        min_k = k;
        min_cost = curr_cost;
      }
    }

    // If we found a list of middle nodes with current boxing collector
    int32_t producer_hierarchy_num = producer_parallel_desc.hierarchy()->NumAxes();
    if (min_k >= 0) {
      for (int32_t middle_sbp_id : middle_nodes_[i][j][min_k]) {
        middle_sbps.emplace_back(
            *JUST(SetNdSbpDim(nd_sbp_lists_[middle_sbp_id], producer_hierarchy_num)));
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
  JUST(customized_boxing_collector.Init(logical_blob_desc, producer_parallel_desc));
  JUST(customized_boxing_collector.AskSbpCombination4Same2DPlacement(
      sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc, consumer_parallel_desc,
      /*is_customized=*/true, middle_sbps, diag_node_pos, compute_cost));
  return Maybe<void>::Ok();
}

// Ask for sbp combination with different hierarchies and placements
Maybe<void> BoxingCollector::AskSbpCombination4DiffPlacement(
    const NdSbp& sbp_producer, const NdSbp& sbp_consumer, const BlobDesc& logical_blob_desc,
    const ParallelDesc& producer_parallel_desc, const ParallelDesc& consumer_parallel_desc,
    bool is_customized, std::vector<NdSbp>& middle_sbps, int32_t* diag_node_pos,
    bool compute_cost) {
  middle_sbps.clear();
  // Find the 2D sbp id
  int32_t i = FindId4NdSbp(sbp_producer);
  int32_t j = FindId4NdSbp(sbp_consumer);
  // Different placements: [2, 3] vs 5, or [3, 2] vs [2, 2], or cpu vs cuda
  // Different hierarchies: [2, 3] vs 5, or [4, 3] vs [6, 2]
  bool same_placement = producer_parallel_desc.EqualsIgnoringHierarchy(consumer_parallel_desc);
  // Dealing with 2D sbp
  if (i >= 0 && j >= 0) {
    // Pure copy between machines and devices
    if (i == j && (*producer_parallel_desc.hierarchy() == *consumer_parallel_desc.hierarchy())) {
      return Maybe<void>::Ok();
    }
    if (same_placement) {
      // Different hierarchies
      CHECK_OR_RETURN(diag_node_diff_hierarchy_.size() > 0)
          << "Have not initialized the combination table for different hierarchies yet! "
             "Please run JUST(GenerateCombination4DiffHierarchy(this, this)); "
             "before Asking sbp combination for different parallel description.";
      if (JUST(Ask1Combination4DiffPlacement(
              sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc,
              consumer_parallel_desc, is_customized, middle_sbps, diag_node_pos, compute_cost, this,
              this, diag_node_diff_hierarchy_[i][j]))) {
        return Maybe<void>::Ok();
      }
    } else {
      // Different placements
      CHECK_OR_RETURN(diag_node_diff_placement_.size() > 0)
          << "Have not initialized the combination table for different hierarchies yet! "
             "Please run JUST(GenerateCombination4DiffPlacement(this, this)); "
             "before Asking sbp combination for different parallel description.";
      if (JUST(Ask1Combination4DiffPlacement(
              sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc,
              consumer_parallel_desc, is_customized, middle_sbps, diag_node_pos, compute_cost, this,
              this, diag_node_diff_placement_[i][j]))) {
        return Maybe<void>::Ok();
      }
    }
  }
  // Customized boxing collector and try the algorithm again
  if (is_customized) {
    CHECK_OR_RETURN(compute_cost) << "Boxing does not support " << NdSbpToString(sbp_producer)
                                  << "[hierarchy: " << *producer_parallel_desc.hierarchy()
                                  << "] -> " << NdSbpToString(sbp_consumer)
                                  << "[hierarchy: " << *consumer_parallel_desc.hierarchy()
                                  << "] for blob shape: " << logical_blob_desc.shape();
    return Maybe<void>::Ok();
  }
  // Customize boxing collector for producer
  BoxingCollector customized_boxing_collector_producer;
  JUST(customized_boxing_collector_producer.Init(logical_blob_desc, producer_parallel_desc));
  // Customize boxing collector for consumer
  BoxingCollector customized_boxing_collector_consumer;
  JUST(customized_boxing_collector_consumer.Init(logical_blob_desc, consumer_parallel_desc));

  std::vector<std::vector<int32_t>> diag_nodes;
  // Generate the combination table for different hierarchies or placements
  if (same_placement) {
    JUST(customized_boxing_collector_producer.Generate1Combination4DiffHierarchy(
        customized_boxing_collector_producer.FindId4NdSbp(sbp_producer),
        customized_boxing_collector_consumer.FindId4NdSbp(sbp_consumer),
        &customized_boxing_collector_producer, &customized_boxing_collector_consumer, diag_nodes));
  } else {
    // Compute the cost while transferring a 1D sbp between different placements
    std::vector<std::vector<double>> cost_4_diff_placement;
    JUST(ComputeCostFor1DSbpDiffPlacement(logical_blob_desc, producer_parallel_desc,
                                          consumer_parallel_desc, cost_4_diff_placement));

    JUST(customized_boxing_collector_producer.Generate1Combination4DiffPlacement(
        customized_boxing_collector_producer.FindId4NdSbp(sbp_producer),
        customized_boxing_collector_consumer.FindId4NdSbp(sbp_consumer),
        &customized_boxing_collector_producer, &customized_boxing_collector_consumer,
        cost_4_diff_placement, diag_nodes));
  }

  JUST(customized_boxing_collector_producer.Ask1Combination4DiffPlacement(
      sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc, consumer_parallel_desc,
      /*is_customized=*/true, middle_sbps, diag_node_pos, compute_cost,
      &customized_boxing_collector_producer, &customized_boxing_collector_consumer, diag_nodes));
  return Maybe<void>::Ok();
}

// Generate the transfer rule for one combination with different hierarchies on the same
// placement. id_producer -> id_consumer.
Maybe<void> BoxingCollector::Generate1Combination4DiffHierarchy(
    int32_t id_producer, int32_t id_consumer, BoxingCollector* boxing_collector_producer,
    BoxingCollector* boxing_collector_consumer, std::vector<std::vector<int32_t>>& diag_nodes) {
  // Number of 1d sbp
  int32_t m = id2sbp_parallel_.size();

  // Search the path that contains one of the diagonal sbp

  // minimum number of node
  int32_t min_path_length = 100;
  // minimum cost
  double min_cost = GetValidMaxCopyCost();

  for (int32_t id_1d = 0; id_1d < m; id_1d++) {
    // We do not support [2, 3]: (S0, S1) -> [6]: S0 for a tensor with shape (14, 21)
    // Thus, the diagonal node should suit both the hierarchies.
    int32_t diag_producer = boxing_collector_producer->id_1d_2_nd_[id_1d];
    if (diag_producer < 0) { continue; }
    int32_t diag_consumer = boxing_collector_consumer->id_1d_2_nd_[id_1d];
    if (diag_consumer < 0) { continue; }
    // Find the path with minimum number of nodes
    int32_t path_length = 0;
    // Transfer from id_producer to id_2d
    if (boxing_collector_producer->middle_nodes_[id_producer][diag_producer].size() > 0) {
      path_length +=
          boxing_collector_producer->middle_nodes_[id_producer][diag_producer][0].size() + 1;
    } else if (id_producer != diag_producer) {
      path_length++;
    }
    // Transfer from id_2d to id_consumer
    if (boxing_collector_consumer->middle_nodes_[diag_consumer][id_consumer].size() > 0) {
      path_length +=
          boxing_collector_consumer->middle_nodes_[diag_consumer][id_consumer][0].size() + 1;
    } else if (diag_consumer != id_consumer) {
      path_length++;
    }
    // Pick the path with minimum copy cost
    if (path_length <= min_path_length) {
      double curr_cost =
          boxing_collector_producer->minimum_copy_cost_[id_producer][diag_producer]
          + boxing_collector_consumer->minimum_copy_cost_[diag_consumer][id_consumer];

      min_path_length = path_length;
      // Find a candidate with small cost
      if (curr_cost < min_cost * kFloatDeviationPlus) {
        // Find a smaller cost, clear the previous path.
        if (curr_cost < min_cost * kFloatDeviationMinus) {
          min_cost = curr_cost;
          diag_nodes.clear();
        }
        // Add the current diagonal node
        // Asymmetry happens here. We can only store one side of the diagonal node.
        // We do not store diag_consumer
        diag_nodes.push_back({diag_producer, diag_consumer});
      }
    }
  }

  return Maybe<void>::Ok();
}

// Ask for one combination with different hierarchies and placements
Maybe<bool> BoxingCollector::Ask1Combination4DiffPlacement(
    const NdSbp& sbp_producer, const NdSbp& sbp_consumer, const BlobDesc& logical_blob_desc,
    const ParallelDesc& producer_parallel_desc, const ParallelDesc& consumer_parallel_desc,
    bool is_customized, std::vector<NdSbp>& middle_sbps, int32_t* diag_node_pos, bool compute_cost,
    BoxingCollector* boxing_collector_producer, BoxingCollector* boxing_collector_consumer,
    const std::vector<std::vector<int32_t>>& diag_nodes) {
  // Pick the path with minimum storage for the diagonal node
  int32_t id_producer = boxing_collector_producer->FindId4NdSbp(sbp_producer);
  if (id_producer < 0) {
    CHECK_OR_RETURN(compute_cost) << "Source data with shape " << logical_blob_desc.shape()
                                  << " has an invalid sbp " << NdSbpToString(sbp_producer);
    return false;
  }
  int32_t id_consumer = boxing_collector_consumer->FindId4NdSbp(sbp_consumer);
  if (id_consumer < 0) {
    CHECK_OR_RETURN(compute_cost) << "Target data with shape " << logical_blob_desc.shape()
                                  << " has an invalid sbp " << NdSbpToString(sbp_consumer);
    return false;
  }
  middle_sbps.clear();
  // NOTE: For simplicity, We do not dig into those storage cost for the other middle nodes at
  // this moment.
  double min_cost = GetValidMaxCopyCost();
  int32_t producer_hierarchy_num_axes = producer_parallel_desc.hierarchy()->NumAxes();
  int32_t consumer_hierarchy_num_axes = consumer_parallel_desc.hierarchy()->NumAxes();
  int32_t min_diag_producer = -1, min_diag_consumer = -1;
  for (const auto& diag_pair : diag_nodes) {
    Shape logical_shape = logical_blob_desc.shape();
    // We do not check whether such shape is valid under two side of the sbp list in the
    // middle nodes algorithm. Thus, we need to check them here.
    double curr_cost =
        Storage4NdSbp(*JUST(SetNdSbpDim(boxing_collector_producer->nd_sbp_lists_[diag_pair[0]],
                                        producer_hierarchy_num_axes)),
                      logical_shape, *producer_parallel_desc.hierarchy());
    // Check the shape for both producer and consumer.
    logical_shape = logical_blob_desc.shape();
    curr_cost +=
        Storage4NdSbp(*JUST(SetNdSbpDim(boxing_collector_consumer->nd_sbp_lists_[diag_pair[1]],
                                        consumer_hierarchy_num_axes)),
                      logical_shape, *consumer_parallel_desc.hierarchy());
    if (curr_cost < min_cost) {
      min_cost = curr_cost;
      min_diag_producer = diag_pair[0];
      min_diag_consumer = diag_pair[1];
    }
  }

  // Different placements: [2, 3] vs 5, or [3, 2] vs [2, 2], or cpu vs cuda
  // Different hierarchies: [2, 3] vs 5, or [4, 3] vs [6, 2]
  bool diff_placement = !producer_parallel_desc.EqualsIgnoringHierarchy(consumer_parallel_desc);

  // If we found a diagonal middle node with current boxing collector
  if (min_diag_producer >= 0) {
    std::vector<NdSbp> middle_sbps_buffer;
    // Find the middle nodes between the producer and the diagonal node
    if (id_producer != min_diag_producer) {
      JUST(boxing_collector_producer->AskSbpCombination(
          sbp_producer, boxing_collector_producer->nd_sbp_lists_[min_diag_producer],
          logical_blob_desc, producer_parallel_desc, producer_parallel_desc,
          /*is_customized=*/false, middle_sbps_buffer, diag_node_pos, compute_cost));
      // Add the path into middle_sbps
      for (auto& middle_sbp : middle_sbps_buffer) {
        middle_sbps.emplace_back(*JUST(SetNdSbpDim(middle_sbp, producer_hierarchy_num_axes)));
      }
      // If different placement,
      // or the same placement but with 2D hierarchies
      // For example: Oneflow supports [6]: (S0) -> [3, 2]: (S0, S1)
      // but does not support [2, 3]: (S0, S0) -> [3, 2]: (S0, S1)
      if (diff_placement || producer_hierarchy_num_axes > 1) {
        middle_sbps.emplace_back(
            *JUST(SetNdSbpDim(boxing_collector_producer->nd_sbp_lists_[min_diag_producer],
                              producer_hierarchy_num_axes)));
      }
    }
    // If we do not have middle nodes on the consumer side
    *diag_node_pos = middle_sbps.size();
    // Find the middle nodes between the diagonal node and the consumer
    if (id_consumer != min_diag_consumer) {
      JUST(boxing_collector_consumer->AskSbpCombination(
          boxing_collector_consumer->nd_sbp_lists_[min_diag_consumer], sbp_consumer,
          logical_blob_desc, consumer_parallel_desc, consumer_parallel_desc,
          /*is_customized=*/false, middle_sbps_buffer, diag_node_pos, compute_cost));
      // Set the diagonal node position and stop using it as buffer
      *diag_node_pos = middle_sbps.size();
      // If different placement
      if (diff_placement || consumer_hierarchy_num_axes > 1) {
        middle_sbps.emplace_back(
            *JUST(SetNdSbpDim(boxing_collector_consumer->nd_sbp_lists_[min_diag_consumer],
                              consumer_hierarchy_num_axes)));
      }
      // Add the path into middle_sbps
      for (auto& middle_sbp : middle_sbps_buffer) {
        middle_sbps.emplace_back(*JUST(SetNdSbpDim(middle_sbp, consumer_hierarchy_num_axes)));
      }
    }
    return true;
  }
  return false;
}

// Generate the transfer rule for one combination with different placements
// id_producer -> id_consumer.
Maybe<void> BoxingCollector::Generate1Combination4DiffPlacement(
    int32_t id_producer, int32_t id_consumer, BoxingCollector* boxing_collector_producer,
    BoxingCollector* boxing_collector_consumer,
    const std::vector<std::vector<double>>& cost_4_diff_placement,
    std::vector<std::vector<int32_t>>& diag_nodes) {
  // Number of 1d sbp
  int32_t m = id2sbp_parallel_.size();
  // minimum number of node
  int32_t min_path_length = 100;
  // minimum cost
  double min_cost = GetValidMaxCopyCost();

  // Search the path that contains two of the diagonal sbp
  // From the producer to the first diagonal node
  for (int32_t id_1d_producer = 0; id_1d_producer < m; id_1d_producer++) {
    // We do not support [2, 3]: (S0, S1) -> [6]: S0 for a tensor with shape (14, 21)
    // Thus, the diagonal node should suit both the hierarchies.
    int32_t diag_producer = boxing_collector_producer->id_1d_2_nd_[id_1d_producer];
    if (diag_producer < 0
        || boxing_collector_producer->minimum_copy_cost_[id_producer][diag_producer]
               > GetValidMaxCopyCost()) {
      continue;
    }
    // Find the path with minimum number of nodes
    int32_t path_length = 0;
    // Transfer from id_producer to diag_producer
    if (boxing_collector_producer->middle_nodes_[id_producer][diag_producer].size() > 0) {
      path_length +=
          boxing_collector_producer->middle_nodes_[id_producer][diag_producer][0].size() + 1;
    } else if (id_producer != diag_producer) {
      path_length++;
    }
    // pruning
    if (path_length > min_path_length) { continue; }

    // From the second diagonal node to the consumer
    for (int32_t id_1d_consumer = 0; id_1d_consumer < m; id_1d_consumer++) {
      int32_t diag_consumer = boxing_collector_consumer->id_1d_2_nd_[id_1d_consumer];
      // The diagonal sbp is not supported or no paths exist from the diagonal sbp to the
      // consumer or between the two diagonal sbps.
      if (diag_consumer < 0
          || boxing_collector_consumer->minimum_copy_cost_[diag_consumer][id_consumer]
                 > GetValidMaxCopyCost()
          || cost_4_diff_placement[id_1d_producer][id_1d_consumer] > GetValidMaxCopyCost()) {
        continue;
      }

      // Transfer from diag_consumer to id_consumer
      int32_t curr_path_length = path_length;
      if (boxing_collector_consumer->middle_nodes_[diag_consumer][id_consumer].size() > 0) {
        curr_path_length +=
            boxing_collector_consumer->middle_nodes_[diag_consumer][id_consumer][0].size() + 1;
      } else if (diag_consumer != id_consumer) {
        curr_path_length++;
      }
      // Pick the path with minimum copy cost
      if (curr_path_length <= min_path_length) {
        double curr_cost =
            boxing_collector_producer->minimum_copy_cost_[id_producer][diag_producer]
            + cost_4_diff_placement[id_1d_producer][id_1d_consumer]
            + boxing_collector_consumer->minimum_copy_cost_[diag_consumer][id_consumer];

        min_path_length = curr_path_length;
        // Find a candidate with small cost
        if (curr_cost < min_cost * 1.0000001) {
          // Find a smaller cost, clear the previous path.
          if (curr_cost < min_cost * 0.9999999) {
            min_cost = curr_cost;
            diag_nodes.clear();
          }
          // Add the current diagonal node
          // Asymmetry happens here. We can only store one side of the diagonal node.
          // We do not store diag_consumer
          diag_nodes.push_back({diag_producer, diag_consumer});
        }
      }
    }
  }

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
      nd_sbp_universe_[nd_sbp_lists_[nd_sbp_lists_.size() - 1]] = middle_sbp_id;
      nd_sbp_universe_.erase(nd_sbp_lists_[middle_sbp_id]);
      nd_sbp_lists_[middle_sbp_id] = nd_sbp_lists_[nd_sbp_lists_.size() - 1];
      nd_sbp_lists_.pop_back();
    }
  }
  return Maybe<void>::Ok();
}

// Ask for sbp combination for general basic communication
Maybe<void> BoxingCollector::AskSbpCombination4GeneralBasicCommunication(
    const NdSbp& sbp_producer, const NdSbp& sbp_consumer, const BlobDesc& logical_blob_desc,
    const ParallelDesc& producer_parallel_desc, const ParallelDesc& consumer_parallel_desc,
    std::vector<NdSbp>& middle_sbps, int32_t* diag_node_pos) {
  // (P, X) -> (B, X) || (X , P) -> (X, B), X is any SBP
  // One step transfer, at most 50% reduction in the transfer cost, do not use middle nodes
  if (producer_parallel_desc == consumer_parallel_desc
      && producer_parallel_desc.hierarchy()->NumAxes() == 2
      && (sbp_producer.sbp_parallel(0) == sbp_consumer.sbp_parallel(0)
          || sbp_producer.sbp_parallel(1) == sbp_consumer.sbp_parallel(1))) {
    return Maybe<void>::Ok();
  }

  // Not enough gain in transfer cost, do not use middle nodes
  int32_t partial_ratio4producer = PartialRatio4Producer(sbp_producer, producer_parallel_desc);
  int32_t broadcast_ratio4consumer = BroadcastRatio4Consumer(sbp_consumer, consumer_parallel_desc);
  if (2 * (partial_ratio4producer + broadcast_ratio4consumer)
      >= partial_ratio4producer * broadcast_ratio4consumer) {
    return Maybe<void>::Ok();
  }

  bool close2producer = true;
  if (producer_parallel_desc.parallel_num() == consumer_parallel_desc.parallel_num()) {
    // Get close to the one with more splits
    close2producer = TotalNumSplit(sbp_producer, producer_parallel_desc)
                     > TotalNumSplit(sbp_consumer, consumer_parallel_desc);
  } else {
    // Get close to the one with more machines
    close2producer = producer_parallel_desc.parallel_num() > consumer_parallel_desc.parallel_num();
  }
  // Get the contiguous sbp
  if (close2producer) {
    JUST(AskCloseAllSplitSbp(sbp_producer, producer_parallel_desc, logical_blob_desc, middle_sbps));
    *diag_node_pos = 1;
  } else {
    JUST(AskCloseAllSplitSbp(sbp_consumer, consumer_parallel_desc, logical_blob_desc, middle_sbps));
    *diag_node_pos = 0;
  }
  return Maybe<void>::Ok();
}

// Ask for a all-split sbp which is close to the original one
Maybe<void> BoxingCollector::AskCloseAllSplitSbp(const NdSbp& nd_sbp,
                                                 const ParallelDesc& parallel_desc,
                                                 const BlobDesc& logical_blob_desc,
                                                 std::vector<NdSbp>& middle_sbps) {
  Shape remain_shape = logical_blob_desc.shape();
  Shape rest_split_shape = logical_blob_desc.shape();
  int32_t dim_shape = remain_shape.NumAxes();
  // Initialize the remains and splitting
  // logical_blob_desc.shape() == remain_shape .* rest_split_shape;
  for (int32_t i = 0; i < dim_shape; i++) { rest_split_shape.Set(i, 1); }
  for (int32_t sbp_id = 0; sbp_id < nd_sbp.sbp_parallel_size(); sbp_id++) {
    const auto& sbp = nd_sbp.sbp_parallel(sbp_id);
    if (sbp.has_split_parallel()) {
      int32_t axis = sbp.split_parallel().axis();
      int32_t split_num = parallel_desc.hierarchy()->At(sbp_id);
      remain_shape.Set(axis, remain_shape.At(axis) / split_num);
      rest_split_shape.Set(axis, rest_split_shape.At(axis) * split_num);
    }
  }
  // Get the contiguous sbp
  NdSbp new_sbp = nd_sbp;
  for (int32_t sbp_id = 0; sbp_id < nd_sbp.sbp_parallel_size(); sbp_id++) {
    const auto& sbp = nd_sbp.sbp_parallel(sbp_id);
    int32_t split_num = parallel_desc.hierarchy()->At(sbp_id);
    if (sbp.has_split_parallel()) {
      int32_t axis = sbp.split_parallel().axis();
      // split shape is the total splitting number starting from sbp_id to the end
      rest_split_shape.Set(axis, rest_split_shape.At(axis) / split_num);
    } else {
      // change P or B to S(axis)
      int32_t axis = -1;
      // 4096 is large enough, we might not have that much devices
      int32_t min_split_num = 4096;
      // We need to pick a suitable axis
      for (int32_t i = 0; i < remain_shape.NumAxes(); i++) {
        if (remain_shape.At(i) % split_num == 0) {
          if (rest_split_shape.At(i) < min_split_num) {
            // Pick the axis with smallest splitting number among the rest of the sbp
            min_split_num = rest_split_shape.At(i);
            axis = i;
          }
        }
      }
      // P, B -> S(axis)
      if (axis >= 0) {
        new_sbp.mutable_sbp_parallel(sbp_id)->mutable_split_parallel()->set_axis(axis);
        remain_shape.Set(axis, remain_shape.At(axis) / split_num);
      } else {
        // Can not find a suitable contiguous sbp
        return Maybe<void>::Ok();
      }
    }
  }
  // Add the new sbp into the middle node lists
  middle_sbps.emplace_back(new_sbp);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
