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

#include "boxing_collector.h"
#include "sbp_node.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace {
void DfsSetNdSbp(std::vector<::oneflow::cfg::SbpParallel>& id2SbpParallel, int32_t depth,
                 int32_t max_depth, cfg::NdSbp& nd_sbp, std::vector<cfg::NdSbp>& nd_sbp_lists) {
  if (depth == max_depth) {
    nd_sbp_lists.push_back(nd_sbp);
  } else {
    for (int32_t i = 0; i < id2SbpParallel.size(); i++) {
      *nd_sbp.mutable_sbp_parallel(depth) = id2SbpParallel[i];
      DfsSetNdSbp(id2SbpParallel, depth + 1, max_depth, nd_sbp, nd_sbp_lists);
    }
  }
}
}  // namespace

// Construct a boxing collector with given operator graph
BoxingCollector::BoxingCollector(const OpGraph& op_graph) {
  CollectUniverse(op_graph);
  GenerateCombination();
}

// Collect all the possible Sbp Parallel from an OpGraph
void BoxingCollector::CollectUniverse(const OpGraph& op_graph) {
  op_graph.ForEachNode([&](const OpNode* node) -> void {
    const auto& nd_sbp_sig = node->nd_sbp_signature();
    for (const auto& pair : nd_sbp_sig.bn_in_op2nd_sbp()) {
      for (const auto& sbp : pair.second.sbp_parallel()) {
        if (SbpParallelUniverse.find(sbp) == SbpParallelUniverse.end()) {
          int32_t curr_size = SbpParallelUniverse.size();
          SbpParallelUniverse[sbp] = curr_size;
          id2SbpParallel.push_back(sbp);
        }
      }
    }
  });
}

// Generate the transfer rule for different combinations and hierarchies
Maybe<void> BoxingCollector::GenerateCombination() {
  // 1D sbp does not support S->P. But it seems that we do not need to deal with it for now.
  // And we do not have 3D sbp or higher dimension.
  int32_t hierarchy_num = 2;

  // Generate possible nd_sbp lists
  cfg::NdSbp nd_sbp;
  for (int32_t dim_sbp = 0; dim_sbp < hierarchy_num; dim_sbp++) { nd_sbp.add_sbp_parallel(); }
  DfsSetNdSbp(id2SbpParallel, 0, hierarchy_num, nd_sbp, nd_sbp_lists);
  // other parameters
  // To be noted that the performance of this function are all the same with different hierarchy
  Shape hierarchy44({4, 4});
  std::shared_ptr<Shape> in_hierarchy = std::make_shared<Shape>(hierarchy44);
  double logical_blob_size = 1024.0;
  // Store the origin transfer cost information
  int32_t n = nd_sbp_lists.size();
  minimum_copy_cost.resize(n);
  middle_nodes.resize(n);
  for (int32_t i = 0; i < n; i++) {
    minimum_copy_cost[i].resize(n);
    middle_nodes[i].resize(n);
    for (int32_t j = 0; j < n; j++) {
      minimum_copy_cost[i][j] = JUST(auto_parallel::ComputCopyCostBetweenNdSbp(
          nd_sbp_lists[i], nd_sbp_lists[j], logical_blob_size, in_hierarchy, in_hierarchy));
    }
  }
  // test debug
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "===================origin copy cost==================" << std::endl;
    // Print the origin copy cost table
    std::cout << "Cost\t";
    for (int32_t j = 0; j < n; j++) { std::cout << NdSbpParallelToString(nd_sbp_lists[j]) << "\t"; }
    std::cout << std::endl;
    for (int32_t i = 0; i < n; i++) {
      std::cout << NdSbpParallelToString(nd_sbp_lists[i]) << "\t";
      for (int32_t j = 0; j < n; j++) {
        if (minimum_copy_cost[i][j] > cut_cost) {
          std::cout << "X\t";
        } else {
          std::cout << minimum_copy_cost[i][j] << "\t";
        }
      }
      std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Original Copy Cost" << std::endl;
    std::cout << "logical blob size: " << logical_blob_size << std::endl;
    std::cout << "hierarchy: " << *in_hierarchy << std::endl;
  }

  for (int32_t middle_node_num = 1; middle_node_num <= hierarchy_num; middle_node_num++) {
    int32_t middle_node_num_ik = middle_node_num - 1;

    for (int32_t i = 0; i < n; i++) {
      for (int32_t j = 0; j < n; j++) {
        if (minimum_copy_cost[i][j] < cut_cost) { continue; }
        // Compute the smallest transfer cost
        // k is the middle node, i -> k -> j
        for (int32_t k = 0; k < n; k++) {
          if (k == j || k == i) { continue; }
          if (middle_nodes[k][j].size() > 0) { continue; }
          double curr_copy_cost = minimum_copy_cost[i][k] + minimum_copy_cost[k][j];
          if (curr_copy_cost < minimum_copy_cost[i][j]) {
            minimum_copy_cost[i][j] = curr_copy_cost;
          }
        }
        // If the minimum copy cost remians infinity, adding one middle node does not make it.
        if (minimum_copy_cost[i][j] > cut_cost) { continue; }
        // Find those middle nodes
        for (int32_t k = 0; k < n; k++) {
          // Not allow i -> i -> j or i -> j -> j.
          if (k == j || k == i) { continue; }
          // We add middle nodes one by one
          // Thus, we allow multiple nodes from i to k but we only accept 1 step from k to j.
          // i -> ? -> k -> j
          if (middle_nodes[k][j].size() == 0) {
            // To avoid multiple counting and bugs, the number of middle nodes between i and k
            // must be exactly middle_node_num_ik, which is (middle_node_num - 1)
            if (middle_node_num_ik) {
              if (middle_nodes[i][k].size() == 0
                  || middle_nodes[i][k][0].size() != middle_node_num_ik) {
                continue;
              }
            } else {
              if (middle_nodes[i][k].size() > 0) { continue; }
            }
            // Now we start to judge if the edge have a minimum cost
            double curr_copy_cost = minimum_copy_cost[i][k] + minimum_copy_cost[k][j];
            if (curr_copy_cost < cut_cost && curr_copy_cost < minimum_copy_cost[i][j] * 1.0000001) {
              // i -> ? -> k
              if (middle_nodes[i][k].size() > 0) {
                // We have multiple choices going from i to k
                for (const auto& middle_node_ik : middle_nodes[i][k]) {
                  middle_nodes[i][j].push_back(middle_node_ik);
                  middle_nodes[i][j][middle_nodes[i][j].size() - 1].push_back(k);
                }
              } else {
                // We only need one middle node k to reach j from i
                middle_nodes[i][j].push_back({k});
              }
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
  // test debug
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "===================minimum copy cost==================" << std::endl;
    // other parameters
    // To be noted that the performance of this function are all the same with different hierarchy
    Shape hierarchy44({4, 4});
    std::shared_ptr<Shape> in_hierarchy = std::make_shared<Shape>(hierarchy44);
    double logical_blob_size = 1024.0;
    int32_t n = nd_sbp_lists.size();
    // Print the origin copy cost table
    std::cout << "Cost\t";
    for (int32_t j = 0; j < n; j++) { std::cout << NdSbpParallelToString(nd_sbp_lists[j]) << "\t"; }
    std::cout << std::endl;
    for (int32_t i = 0; i < n; i++) {
      std::cout << NdSbpParallelToString(nd_sbp_lists[i]) << "\t";
      for (int32_t j = 0; j < n; j++) {
        if (minimum_copy_cost[i][j] > cut_cost) {
          std::cout << "X\t";
        } else {
          std::cout << minimum_copy_cost[i][j] << "\t";
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
    for (int32_t j = 0; j < n; j++) { std::cout << NdSbpParallelToString(nd_sbp_lists[j]) << "\t"; }
    std::cout << std::endl;
    for (int32_t i = 0; i < n; i++) {
      std::cout << NdSbpParallelToString(nd_sbp_lists[i]) << "\t";
      for (int32_t j = 0; j < n; j++) {
        if (minimum_copy_cost[i][j] > cut_cost) {
          std::cout << "X";
        } else if (middle_nodes[i][j].size() > 0) {
          for (int32_t k = 0; k < middle_nodes[i][j].size(); k++) {
            std::cout << NdSbpParallelToString(nd_sbp_lists[middle_nodes[i][j][k][0]]);
            for (int32_t l = 1; l < middle_nodes[i][j][k].size(); l++) {
              std::cout << "->" << NdSbpParallelToString(nd_sbp_lists[middle_nodes[i][j][k][l]]);
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

    std::cout << "================================================" << std::endl;
  }
}
}  // namespace oneflow