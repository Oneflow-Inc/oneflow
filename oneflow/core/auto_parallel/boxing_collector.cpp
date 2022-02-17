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
#include "oneflow/core/auto_parallel/boxing_collector.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/lazy_mode.h"

namespace oneflow {

namespace {
void DfsSetNdSbp(std::vector<::oneflow::cfg::SbpParallel>& id2SbpParallel, int32_t depth,
                 int32_t max_depth, cfg::NdSbp& nd_sbp, std::vector<cfg::NdSbp>& nd_sbp_lists_,
                 std::unordered_map<::oneflow::cfg::NdSbp, int32_t>& NdSbpUniverse_) {
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

// If an nd sbp can be converted to a 1d sbp.
bool Is1dSbp(const cfg::NdSbp& nd_sbp) {
  for (int32_t i = 1; i < nd_sbp.sbp_parallel_size(); i++) {
    if (nd_sbp.sbp_parallel(0) != nd_sbp.sbp_parallel(i)) { return false; }
  }
  return true;
}

// Let a nd sbp be consistent with the given hierarchy number
Maybe<cfg::NdSbp> SetNdSbpDim(cfg::NdSbp nd_sbp, int32_t hierarchy_num) {
  // Do not need to change
  if (nd_sbp.sbp_parallel_size() == hierarchy_num) { return nd_sbp; }
  // (S0, S0) -> S0
  if (hierarchy_num == 1) {
    CHECK_OR_RETURN(Is1dSbp(nd_sbp))
        << NdSbpToString(nd_sbp) << " can not be converted to a 1d sbp!";
    cfg::NdSbp new_sbp;
    new_sbp.add_sbp_parallel();
    *new_sbp.mutable_sbp_parallel(0) = nd_sbp.sbp_parallel(0);
    return new_sbp;
  }
  // S0 -> (S0, S0)
  cfg::NdSbp new_sbp;
  for (int32_t i = 0; i < hierarchy_num; i++) {
    new_sbp.add_sbp_parallel();
    *new_sbp.mutable_sbp_parallel(i) = nd_sbp.sbp_parallel(0);
  }
  return new_sbp;
}

}  // namespace

// A constructor with init, designed for uncustomized boxing collector
BoxingCollector::BoxingCollector(int32_t max_axis) { CHECK_JUST(Init(max_axis)); }

// Construct a boxing collector with given maximum number of axis
Maybe<void> BoxingCollector::Init(int32_t max_axis) {
  // Set up at least two split for op graph.
  // For a negative example: Resnet50 only have B, P, S(0)
  CollectUniverse(max_axis);
  GenerateNdSbpList(2);
  GenerateMap1d2nd();
  JUST(GenerateCombination4SamePlacement(3));
  JUST(GenerateCombination4DiffHierarchy(3));
  JUST(GenerateCombination4DiffHierarchy(3));
  return Maybe<void>::Ok();
}

// Init with given blob description
Maybe<void> BoxingCollector::Init(const BlobDesc& logical_blob_desc,
                                  const ParallelDesc& parallel_desc) {
  CollectUniverse(logical_blob_desc.shape().NumAxes());
  GenerateNdSbpList(parallel_desc.hierarchy()->NumAxes());
  // Filter out unsuitable middle nodes before computing minimum cost.
  JUST(FilterNdSbpList4LogicalShape(logical_blob_desc, *parallel_desc.hierarchy()));
  JUST(GenerateCombination4SamePlacement(5));
  return Maybe<void>::Ok();
}

// Collect Sbp Parallel
void BoxingCollector::CollectUniverse(const cfg::SbpParallel& sbp) {
  if (SbpParallelUniverse_.find(sbp) == SbpParallelUniverse_.end()) {
    int32_t curr_size = SbpParallelUniverse_.size();
    SbpParallelUniverse_[sbp] = curr_size;
    id2SbpParallel_.push_back(sbp);
  }
}

// Find corresponding id for Nd sbp
int32_t BoxingCollector::FindId4NdSbp(const cfg::NdSbp& nd_sbp) {
  if (nd_sbp.sbp_parallel_size() == 1) {
    const auto& it_nd_sbp = SbpParallelUniverse_.find(nd_sbp.sbp_parallel(0));
    if (it_nd_sbp != SbpParallelUniverse_.end()) {
      return id_1d_2_2d_[it_nd_sbp->second];
    } else {
      return -1;
    }
  } else {
    const auto& it_nd_sbp = NdSbpUniverse_.find(nd_sbp);
    if (it_nd_sbp != NdSbpUniverse_.end()) {
      return it_nd_sbp->second;
    } else {
      return -1;
    }
  }
}

// Set default Sbp list
void BoxingCollector::CollectUniverse(int32_t max_axis) {
  cfg::SbpParallel sbp;
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

  // Generate possible nd_sbp lists
  cfg::NdSbp nd_sbp;
  for (int32_t dim_sbp = 0; dim_sbp < hierarchy_num; dim_sbp++) { nd_sbp.add_sbp_parallel(); }
  DfsSetNdSbp(id2SbpParallel_, 0, hierarchy_num, nd_sbp, nd_sbp_lists_, NdSbpUniverse_);
}

// Generate the map from 1d sbp to 2d sbp
void BoxingCollector::GenerateMap1d2nd() {
  // Number of 1d sbp
  int32_t m = id2SbpParallel_.size();

  // Generate the id Map from 1d sbp to 2d sbp
  int32_t hierarchy_num = 2;
  cfg::NdSbp nd_sbp;
  for (int32_t dim_sbp = 0; dim_sbp < hierarchy_num; dim_sbp++) { nd_sbp.add_sbp_parallel(); }
  id_1d_2_2d_.resize(m, -1);
  for (int32_t id_1d = 0; id_1d < m; id_1d++) {
    for (int32_t dim_sbp = 0; dim_sbp < hierarchy_num; dim_sbp++) {
      *nd_sbp.mutable_sbp_parallel(dim_sbp) = id2SbpParallel_[id_1d];
    }
    // NOTE: The 2d sbp might be filtered out already.
    const auto& it_ = NdSbpUniverse_.find(nd_sbp);
    if (it_ != NdSbpUniverse_.end()) { id_1d_2_2d_[id_1d] = it_->second; }
  }
}

// Generate the transfer rule for different combinations with the same hierarchie
Maybe<void> BoxingCollector::GenerateCombination4SamePlacement(int32_t max_middle_node_num) {
  // other parameters
  // NOTE: The performance of this function are all the same with different hierarchy
  int32_t kWorldSize = GlobalProcessCtx::WorldSize();
  Shape hierarchy44({4 * kWorldSize, 4 * kWorldSize});
  std::shared_ptr<Shape> in_hierarchy = std::make_shared<Shape>(hierarchy44);
  auto in_parallel_desc = JUST(ParallelDesc::New(
      "cpu", {"0:0-" + std::to_string(hierarchy44.elem_cnt() - 1)}, in_hierarchy));
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

// Generate the transfer rule for different combinations with different hierarchies on the same
// placement
Maybe<void> BoxingCollector::GenerateCombination4DiffHierarchy(
    BoxingCollector* boxing_collector_producer, BoxingCollector* boxing_collector_consumer) {
  // Store the boxing collector pointer

  // Search the path that contains one of the diagonal sbp
  int32_t n = nd_sbp_lists_.size();
  diag_node_diff_hierarchy_.resize(n);
  for (int32_t i = 0; i < n; i++) {
    diag_node_diff_hierarchy_[i].resize(n);
    for (int32_t j = 0; j < n; j++) {
      Generate1Combination4DiffHierarchy(i, j, boxing_collector_producer, boxing_collector_consumer,
                                         diag_node_diff_hierarchy_[i][j]);
    }
  }

  return Maybe<void>::Ok();
}

// Generate the transfer rule for different combinations with different placements
Maybe<void> BoxingCollector::GenerateCombination4DiffPlacement(int32_t max_middle_node_num) {
  // Number of 1d sbp
  int32_t m = id2SbpParallel_.size();

  // Other parameters
  int32_t kWorldSize = GlobalProcessCtx::WorldSize();
  BlobDesc blob_desc({16, 16, 16, 16}, DataType::kInt8, /*is_dynamic=*/false);
  // Virtual placements before transfer
  Shape in_hierarchy44({4 * kWorldSize, 4 * kWorldSize});
  std::shared_ptr<Shape> in_hierarchy = std::make_shared<Shape>(in_hierarchy44);
  auto in_parallel_desc = JUST(ParallelDesc::New(
      "cuda", {"0:0-" + std::to_string(in_hierarchy44.elem_cnt() - 1)}, in_hierarchy));
  // Virtual placements after transfer
  Shape out_hierarchy44({4 * kWorldSize, 4 * kWorldSize});
  std::shared_ptr<Shape> out_hierarchy = std::make_shared<Shape>(out_hierarchy44);
  auto out_parallel_desc = JUST(ParallelDesc::New(
      "cpu", {"0:0-" + std::to_string(out_hierarchy44.elem_cnt() - 1)}, out_hierarchy));

  // The cost for transferring a 1D sbp between different placements
  std::vector<std::vector<double>> cost_4_diff_placement;
  // Compute the cost while transferring a 1D sbp between different placements
  cost_4_diff_placement.resize(m);
  for (int32_t id_1d_producer = 0; id_1d_producer < m; id_1d_producer++) {
    cost_4_diff_placement[id_1d_producer].resize(m, GetMaxVal<float>());
    int32_t diag_producer = id_1d_2_2d_[id_1d_producer];
    if (diag_producer < 0) { continue; }

    for (int32_t id_1d_consumer = 0; id_1d_consumer < m; id_1d_consumer++) {
      int32_t diag_consumer = id_1d_2_2d_[id_1d_consumer];
      if (diag_consumer < 0) { continue; }
      cost_4_diff_placement[id_1d_producer][id_1d_consumer] = JUST(ComputeLazyCopyCostBetweenNdSbp(
          nd_sbp_lists_[id_1d_producer], nd_sbp_lists_[diag_consumer], blob_desc, *in_parallel_desc,
          *out_parallel_desc, false));
    }
  }

  // Search the path that contains two of the diagonal sbp
  int32_t n = nd_sbp_lists_.size();
  diag_node_diff_placement_.resize(n);
  for (int32_t i = 0; i < n; i++) {
    diag_node_diff_placement_[i].resize(n);
    for (int32_t j = 0; j < n; j++) {
      // minimum number of node
      int32_t min_path_length = 100;
      // minimum cost
      double min_cost = GetMaxVal<float>();

      // From the producer to the first diagonal node
      for (int32_t id_1d_producer = 0; id_1d_producer < m; id_1d_producer++) {
        int32_t diag_producer = id_1d_2_2d_[id_1d_producer];
        // The diagonal sbp is not supported or no paths exist from the producer to the diagonal
        // sbp.
        if (diag_producer < 0 || minimum_copy_cost_[i][diag_producer] > GetValidMaxCopyCost()) {
          continue;
        }
        // Find the path with minimum number of nodes
        int32_t path_length = 0;
        // Transfer from i to id_2d
        if (middle_nodes_[i][diag_producer].size() > 0) {
          path_length += middle_nodes_[i][diag_producer][0].size() + 1;
        } else if (i != diag_producer) {
          path_length++;
        }

        // From the second diagonal node to the consumer
        for (int32_t id_1d_consumer = 0; id_1d_consumer < m; id_1d_consumer++) {
          int32_t diag_consumer = id_1d_2_2d_[id_1d_consumer];
          // The diagonal sbp is not supported or no paths exist from the diagonal sbp to the
          // consumer or between the two diagonal sbps.
          if (diag_consumer < 0 || minimum_copy_cost_[diag_consumer][j] > GetValidMaxCopyCost()
              || cost_4_diff_placement[id_1d_producer][id_1d_consumer] > GetValidMaxCopyCost()) {
            continue;
          }
          // Transfer from id_2d to j
          if (middle_nodes_[diag_consumer][j].size() > 0) {
            path_length += middle_nodes_[diag_consumer][j][0].size() + 1;
          } else if (diag_consumer != j) {
            path_length++;
          }
          // Pick the path with minimum copy cost
          if (path_length <= min_path_length) {
            double curr_cost = minimum_copy_cost_[i][diag_producer]
                               + cost_4_diff_placement[id_1d_producer][id_1d_consumer]
                               + minimum_copy_cost_[diag_consumer][j];

            min_path_length = path_length;
            // Find a candidate with small cost
            if (curr_cost < min_cost * 1.0000001) {
              // Find a smaller cost, clear the previous path.
              if (curr_cost < min_cost * 0.9999999) {
                min_cost = curr_cost;
                diag_node_diff_placement_[i][j].clear();
              }
              // Add the current diagonal node
              // int32_t diag_producer = diag_producer;
              // if (i == diag_producer) { diag_producer = -1; }
              // int32_t diag_consumer = diag_consumer;
              // if (diag_consumer == j) { diag_consumer = -1; }
              diag_node_diff_placement_[i][j].push_back({diag_producer, diag_consumer});
            }
          }
        }
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

    LOG(INFO) << "====================middle nodes for different placement===================="
              << std::endl;

    std::cout << "Middle nodes for different placement\t";
    for (int32_t j = 0; j < n; j++) { std::cout << NdSbpToString(nd_sbp_lists_[j]) << "\t"; }
    std::cout << std::endl;
    for (int32_t i = 0; i < n; i++) {
      std::cout << NdSbpToString(nd_sbp_lists_[i]) << "\t";
      for (int32_t j = 0; j < n; j++) {
        if (diag_node_diff_placement_[i][j].size() > 0) {
          for (int32_t k = 0; k < diag_node_diff_placement_[i][j].size(); k++) {
            std::cout << NdSbpToString(nd_sbp_lists_[diag_node_diff_placement_[i][j][k]]) << "; ";
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
            std::cout << NdSbpToString(nd_sbp_lists_[diag_node_diff_hierarchy_[i][j][k]]) << "; ";
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
Maybe<void> BoxingCollector::AskSbpCombination(
    const cfg::NdSbp& sbp_producer, const cfg::NdSbp& sbp_consumer,
    const BlobDesc& logical_blob_desc, const ParallelDesc& producer_parallel_desc,
    const ParallelDesc& consumer_parallel_desc, bool is_customized,
    std::vector<cfg::NdSbp>& middle_sbps, int32_t* diag_node_pos, bool compute_cost) {
  middle_sbps.clear();
  // Dealing with 1D sbp to 1D sbp
  // Specifically, S -> P.
  if (Is1dSbp(sbp_producer) && Is1dSbp(sbp_consumer)) {
    if (sbp_producer.sbp_parallel(0).has_split_parallel()
        && sbp_consumer.sbp_parallel(0).has_partial_sum_parallel()) {
      // S -> B -> P (Large cost!)
      // TODO: Please implement S -> P directly.

      int32_t hierarchy_size;
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
    return Maybe<void>::Ok();
  }

  // Dealing with 2D sbp
  int32_t i = FindId4NdSbp(sbp_producer);
  int32_t j = FindId4NdSbp(sbp_consumer);

  // Middle nodes algorithm supports transfer for different machines or devices or hierarchies
  if (producer_parallel_desc != consumer_parallel_desc) {
    if (i >= 0 && j >= 0) {
      bool if_same_placement =
          producer_parallel_desc.EqualsIgnoringHierarchy(consumer_parallel_desc);
      std::vector<int32_t>* diag_nodes;
      if (if_same_placement) {
        // Transfer between different hierarchies under the same placement
        // For example, [2, 3]: (S0, S1) -> [3, 2]: (B, S0)
        // [4]: P -> [2, 2]: (S0, S1)
        CHECK_OR_RETURN(diag_node_diff_hierarchy_.size() > 0)
            << "Have not initialzie the combination table for different hierarchies yet! "
               "Please run JUST(GenerateCombination4DiffHierarchy(6, false)); "
               "before Asking sbp combination for different parallel description.";
        diag_nodes = &(diag_node_diff_hierarchy_[i][j]);
      } else {
        // Transfer between different placements
        // For example, [2, 3]: (S0, S1) -> [5]: B
        // [2, 2]: (P, S0) -> [5, 3]: (P, S0)
        CHECK_OR_RETURN(diag_node_diff_placement_.size() > 0)
            << "Have not initialzie the combination table for different placements yet! "
               "Please run JUST(GenerateCombination4DiffHierarchy(6, true)); "
               "before Asking sbp combination for different parallel description.";
        diag_nodes = &(diag_node_diff_placement_[i][j]);
      }

      // Pick the path with minimum storage for the diagonal node
      // NOTE: For simplicity, We do not dig into those storage cost for the other middle nodes at
      // this moment.
      int32_t min_diag_node = -1;
      double min_cost = GetValidMaxCopyCost();
      for (int32_t id_diag_node : (*diag_nodes)) {
        Shape logical_shape = logical_blob_desc.shape();
        // We do not check whether such shape is valid under two side of the sbp list in the
        // middle nodes algorithm. Thus, we need to check them here.
        double curr_cost = Storage4NdSbp(nd_sbp_lists_[id_diag_node], logical_shape,
                                         *producer_parallel_desc.hierarchy())
                           + Storage4NdSbp(nd_sbp_lists_[id_diag_node], logical_shape,
                                           *consumer_parallel_desc.hierarchy());
        if (curr_cost < min_cost) {
          min_cost = curr_cost;
          min_diag_node = id_diag_node;
        }
      }

      // If we found a diagonal middle node with current boxing collector
      if (min_diag_node >= 0) {
        // Find the middle nodes between the producer and the diagonal node
        std::vector<cfg::NdSbp> middle_sbps_buffer;
        JUST(AskSbpCombination(sbp_producer, nd_sbp_lists_[min_diag_node], logical_blob_desc,
                               producer_parallel_desc, producer_parallel_desc,
                               /*is_customized=*/false, middle_sbps_buffer, diag_node_pos,
                               compute_cost));
        // The last node before the 1d sbp chain at the tail
        // For example, producer: (S0, S1) -> middle node 0: (S0, S0) -> middle node 1: (S0, B) ->
        // middle node 2: B -> diagonal node: (S1, S1)
        // The last node before the chain (B -> S1) would be "middle node 1: (S0, B)"".
        // The shrink_producer_id is 1.
        int32_t shrink_producer_id = middle_sbps_buffer.size() - 1;
        // Shrink multiple 1d sbp
        while (shrink_producer_id >= 0 && Is1dSbp(middle_sbps_buffer[shrink_producer_id])) {
          shrink_producer_id--;
        }
        if (shrink_producer_id < 0 && Is1dSbp(sbp_producer)) { shrink_producer_id--; }
        // Use shrink_producer_id to store the node after the earest 1d sbp in the chain,
        // which is "diagonal node: (S1, S1)" in the example.
        shrink_producer_id += 2;

        // Settle down the middle nodes with parallel description from producer
        if (shrink_producer_id > 0) {
          if (shrink_producer_id > middle_sbps_buffer.size()) {
            // shrink nothing
            middle_sbps.insert(middle_sbps.end(), middle_sbps_buffer.begin(),
                               middle_sbps_buffer.end());
            middle_sbps.emplace_back(nd_sbp_lists_[min_diag_node]);
          } else {
            middle_sbps.insert(middle_sbps.end(), middle_sbps_buffer.begin(),
                               middle_sbps_buffer.begin() + shrink_producer_id);
          }
        }

        // Find the middle nodes between the diagonal node and the consumer
        JUST(AskSbpCombination(nd_sbp_lists_[min_diag_node], sbp_consumer, logical_blob_desc,
                               consumer_parallel_desc, consumer_parallel_desc,
                               /*is_customized=*/false, middle_sbps_buffer, diag_node_pos,
                               compute_cost));
        // Settle down the diag_node_pos
        *diag_node_pos = middle_sbps.size();
        // The first node after the 1d sbp chain at the head
        int32_t shrink_consumer_id = 0;
        // Shrink multiple 1d sbp
        while (shrink_consumer_id < middle_sbps_buffer.size()
               && Is1dSbp(middle_sbps_buffer[shrink_producer_id])) {
          shrink_consumer_id++;
        }
        if (shrink_consumer_id >= middle_sbps_buffer.size() && Is1dSbp(sbp_consumer)) {
          shrink_consumer_id++;
        }
        // Use shrink_producer_id to store the last 1d sbp in the chain
        shrink_consumer_id--;
        // Settle down the middle nodes with parallel description from consumer
        if (shrink_consumer_id < middle_sbps_buffer.size()) {
          if (shrink_consumer_id < 0) {
            // We could shrink one diagonal sbp for the same placement
            // For example: (4): B -> (4): S0 -> (2, 2): (S0, S0) -> (2, 2): (S0, S1)
            // Shrunk: (4):B -> (4): S0 -> (2, 2): (S0, S1)
            if (if_same_placement) {
              // Tail of the current middle sbps
              const cfg::NdSbp* tail_sbp;
              if (middle_sbps.size() > 0) {
                tail_sbp = &(middle_sbps[middle_sbps.size() - 1]);
              } else {
                tail_sbp = &sbp_producer;
              }
              if ((*tail_sbp) != nd_sbp_lists_[min_diag_node]) {
                middle_sbps.emplace_back(nd_sbp_lists_[min_diag_node]);
              }
            } else {
              middle_sbps.emplace_back(nd_sbp_lists_[min_diag_node]);
            }
            // might shrink nothing
            middle_sbps.insert(middle_sbps.end(), middle_sbps_buffer.begin(),
                               middle_sbps_buffer.end());
          } else {
            middle_sbps.insert(middle_sbps.end(), middle_sbps_buffer.begin() + shrink_consumer_id,
                               middle_sbps_buffer.end());
          }
        }
      }
    }

    // Customized boxing collector and try the algorithm again
    CHECK_OR_RETURN(false);
    // BoxingCollector customized_boxing_collector;
    // customized_boxing_collector_producer.CollectUniverse(logical_blob_desc.shape().NumAxes());
    // customized_boxing_collector_producer.GenerateNdSbpList();
    // // Filter out unsuitable middle nodes before computing minimum cost.
    // JUST(customized_boxing_collector_producer.FilterNdSbpList4LogicalShape(logical_blob_desc,
    //                                                               *producer_parallel_desc.hierarchy()));
    // JUST(customized_boxing_collector.GenerateCombination4SamePlacement(5));
    // JUST(customized_boxing_collector.AskSbpCombination(
    //     sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc,
    //     consumer_parallel_desc,
    //     /*is_customized=*/false, middle_sbps, diag_node_pos, compute_cost));
    return Maybe<void>::Ok();
  }
  // Transfer for the same machines, devices and hierarchy.
  const auto& parallel_hierarchy = producer_parallel_desc.hierarchy();
  *diag_node_pos = 0;
  // Dealing with nD sbp, n>2
  if (parallel_hierarchy->NumAxes() > 2) {
    CHECK_OR_RETURN(compute_cost)
        << "Boxing does not support a hierarchy with dimension greater than 2";
    return Maybe<void>::Ok();
  }

  return Maybe<void>::Ok();
}

// Ask for sbp combination with the same 2-D hierarchy and placement
Maybe<void> BoxingCollector::AskSbpCombination4Same2DPlacement(
    const cfg::NdSbp& sbp_producer, const cfg::NdSbp& sbp_consumer,
    const BlobDesc& logical_blob_desc, const ParallelDesc& producer_parallel_desc,
    const ParallelDesc& consumer_parallel_desc, bool is_customized,
    std::vector<cfg::NdSbp>& middle_sbps, int32_t* diag_node_pos, bool compute_cost) {
  CHECK_OR_RETURN(producer_parallel_desc == consumer_parallel_desc)
      << "Producer and consumer have different placements, Please use AskSbpCombination directly";

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
            JUST(SetNdSbpDim(nd_sbp_lists_[middle_sbp_id], producer_hierarchy_num)));
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
  customized_boxing_collector.Init(logical_blob_desc, producer_parallel_desc);
  JUST(customized_boxing_collector.AskSbpCombination4Same2DPlacement(
      sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc, consumer_parallel_desc,
      /*is_customized=*/true, middle_sbps, diag_node_pos, compute_cost));
  return Maybe<void>::Ok();
}

// Ask for sbp combination with different hierarchies on the same placement
Maybe<void> BoxingCollector::AskSbpCombination4DiffHierarchy(
    const cfg::NdSbp& sbp_producer, const cfg::NdSbp& sbp_consumer,
    const BlobDesc& logical_blob_desc, const ParallelDesc& producer_parallel_desc,
    const ParallelDesc& consumer_parallel_desc, bool is_customized,
    std::vector<cfg::NdSbp>& middle_sbps, int32_t* diag_node_pos, bool compute_cost) {
  CHECK_OR_RETURN(producer_parallel_desc.EqualsIgnoringHierarchy(consumer_parallel_desc))
  "Producer and consumer have different placements, Please use AskSbpCombination directly";
  // Find the 2D sbp id
  int32_t i = FindId4NdSbp(sbp_producer);
  int32_t j = FindId4NdSbp(sbp_consumer);
  // Dealing with 2D sbp
  if (i >= 0 && j >= 0) {
    CHECK_OR_RETURN(diag_node_diff_hierarchy_.size() > 0)
        << "Have not initialzie the combination table for different hierarchies yet! "
           "Please run JUST(GenerateCombination4DiffHierarchy(6)); "
           "before Asking sbp combination for different parallel description.";

    Ask1Combination4DiffHierarchy(sbp_producer, sbp_consumer, logical_blob_desc,
                                  producer_parallel_desc, consumer_parallel_desc, is_customized,
                                  middle_sbps, diag_node_pos, compute_cost, this, this,
                                  diag_node_diff_hierarchy_[i][j]);
  }
  // Customized boxing collector and try the algorithm again
  // Customize boxing collector for producer
  BoxingCollector customized_boxing_collector_producer;
  customized_boxing_collector_producer.Init(logical_blob_desc, producer_parallel_desc);
  // Customize boxing collector for consumer
  BoxingCollector customized_boxing_collector_consumer;
  customized_boxing_collector_consumer.Init(logical_blob_desc, consumer_parallel_desc);
  // Generate the combination table for different hierarchies or placements

  JUST(customized_boxing_collector_producer.AskSbpCombination4Same2DPlacement(
      sbp_producer, sbp_consumer, logical_blob_desc, producer_parallel_desc, consumer_parallel_desc,
      /*is_customized=*/true, middle_sbps, diag_node_pos, compute_cost));
  return Maybe<void>::Ok();
}

// Generate the transfer rule for one combination with different hierarchies on the same
// placement. id_producer -> id_consumer.
Maybe<void> BoxingCollector::Generate1Combination4DiffHierarchy(
    int32_t id_producer, int32_t id_consumer, BoxingCollector* boxing_collector_producer,
    BoxingCollector* boxing_collector_consumer, std::vector<std::vector<int32_t>>& diag_nodes) {
  // Number of 1d sbp
  int32_t m = id2SbpParallel_.size();

  // Search the path that contains one of the diagonal sbp

  // minimum number of node
  int32_t min_path_length = 100;
  // minimum cost
  double min_cost = GetMaxVal<float>();

  for (int32_t id_1d = 0; id_1d < m; id_1d++) {
    // We do not support [2, 3]: (S0, S1) -> [6]: S0 for a tensor with shape (14, 21)
    // Thus, the diagonal node should suit both the hierarchies.
    int32_t diag_producer = boxing_collector_producer->id_1d_2_2d_[id_1d];
    if (diag_producer < 0) { continue; }
    int32_t diag_consumer = boxing_collector_consumer->id_1d_2_2d_[id_1d];
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

  return Maybe<void>::Ok();
}

// Ask for one combination with different hierarchies
Maybe<void> BoxingCollector::Ask1Combination4DiffHierarchy(
    const cfg::NdSbp& sbp_producer, const cfg::NdSbp& sbp_consumer,
    const BlobDesc& logical_blob_desc, const ParallelDesc& producer_parallel_desc,
    const ParallelDesc& consumer_parallel_desc, bool is_customized,
    std::vector<cfg::NdSbp>& middle_sbps, int32_t* diag_node_pos, bool compute_cost,
    BoxingCollector* boxing_collector_producer, BoxingCollector* boxing_collector_consumer,
    std::vector<std::vector<int32_t>>& diag_nodes) {
  // Pick the path with minimum storage for the diagonal node
  int32_t id_producer = boxing_collector_producer->FindId4NdSbp(sbp_producer);
  if (id_producer < 0) {
    CHECK_OR_RETURN(compute_cost) << logical_blob_desc.shape() << " has an invalid sbp "
                                  << NdSbpToString(sbp_producer);
  }
  int32_t id_consumer = boxing_collector_consumer->FindId4NdSbp(sbp_consumer);
  if (id_consumer < 0) {
    CHECK_OR_RETURN(compute_cost) << logical_blob_desc.shape() << " has an invalid sbp "
                                  << NdSbpToString(sbp_consumer);
  }
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

  // If we found a diagonal middle node with current boxing collector
  if (min_diag_producer >= 0) {
    std::vector<cfg::NdSbp> middle_sbps_buffer;
    bool diff_sbp4consumer = id_consumer != min_diag_consumer;
    // Find the middle nodes between the producer and the diagonal node
    if (id_producer != min_diag_producer) {
      JUST(boxing_collector_producer->AskSbpCombination(
          sbp_producer, boxing_collector_producer->nd_sbp_lists_[min_diag_producer],
          logical_blob_desc, producer_parallel_desc, producer_parallel_desc,
          /*is_customized=*/false, middle_sbps_buffer, diag_node_pos, compute_cost));
      // Add the path into middle_sbps
      for (auto& middle_sbp : middle_sbps_buffer) {
        middle_sbps.emplace_back(SetNdSbpDim(middle_sbp, producer_hierarchy_num_axes));
      }
      if (diff_sbp4consumer) {
        middle_sbps.emplace_back(
            SetNdSbpDim(boxing_collector_producer->nd_sbp_lists_[min_diag_producer],
                        producer_hierarchy_num_axes));
      }
    }
    // Find the middle nodes between the diagonal node and the consumer
    if (diff_sbp4consumer) {
      JUST(boxing_collector_consumer->AskSbpCombination(
          boxing_collector_consumer->nd_sbp_lists_[min_diag_consumer], sbp_consumer,
          logical_blob_desc, consumer_parallel_desc, consumer_parallel_desc,
          /*is_customized=*/false, middle_sbps_buffer, diag_node_pos, compute_cost));
      // Set the diagonal node position and stop using it as buffer
      *diag_node_pos = middle_sbps.size();
      // Add the path into middle_sbps

      for (auto& middle_sbp : middle_sbps_buffer) {
        middle_sbps.emplace_back(JUST(SetNdSbpDim(middle_sbp, consumer_hierarchy_num_axes)));
      }
    }
  }
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
