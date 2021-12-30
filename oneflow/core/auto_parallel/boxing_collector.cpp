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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/maybe.h"
#include "sbp_node.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "sbp_util.h"

namespace oneflow {

namespace {
void DfsSetNdSbp(std::vector<::oneflow::cfg::SbpParallel>& id2SbpParallel, int32_t depth,
                 int32_t max_depth, cfg::NdSbp& nd_sbp, std::vector<cfg::NdSbp>& nd_sbp_lists,
                 std::unordered_map<::oneflow::cfg::NdSbp, int32_t>& NdSbpUniverse) {
  if (depth == max_depth) {
    NdSbpUniverse[nd_sbp] = nd_sbp_lists.size();
    nd_sbp_lists.push_back(nd_sbp);
  } else {
    for (int32_t i = 0; i < id2SbpParallel.size(); i++) {
      *nd_sbp.mutable_sbp_parallel(depth) = id2SbpParallel[i];
      DfsSetNdSbp(id2SbpParallel, depth + 1, max_depth, nd_sbp, nd_sbp_lists, NdSbpUniverse);
    }
  }
}
}  // namespace

// Construct a boxing collector with given operator graph
void BoxingCollector::Init(const OpGraph& op_graph) {
  minimum_copy_cost.clear();
  middle_nodes.clear();
  nd_sbp_lists.clear();
  SbpParallelUniverse.clear();
  id2SbpParallel.clear();
  // CollectUniverse(2);
  CollectUniverse(op_graph);
  GenerateCombination(2);
}

// Construct a boxing collector with given sbp graph
void BoxingCollector::Init(const auto_parallel::SbpGraph<cfg::NdSbpSignature>& sbp_graph) {
  minimum_copy_cost.clear();
  middle_nodes.clear();
  nd_sbp_lists.clear();
  SbpParallelUniverse.clear();
  id2SbpParallel.clear();
  CollectUniverse(sbp_graph);
  GenerateCombination(2);
}

// Collect all the possible Sbp Parallel from an OpGraph
void BoxingCollector::CollectUniverse(const OpGraph& op_graph) {
  op_graph.ForEachNode(
      [&](const OpNode* node) -> void { CollectUniverse(node->nd_sbp_signature()); });
}

// Collect all the possible Sbp Parallel from an SbpGraph
void BoxingCollector::CollectUniverse(
    const auto_parallel::SbpGraph<cfg::NdSbpSignature>& sbp_graph) {
  for (const auto* sbp_node : sbp_graph.NodeList) { CollectUniverse(sbp_node); }
}

// Collect all the possible Sbp Parallel from a cfg::NdSbpSignature
void BoxingCollector::CollectUniverse(const cfg::NdSbpSignature& nd_sbp_sig) {
  for (const auto& pair : nd_sbp_sig.bn_in_op2nd_sbp()) {
    for (const auto& sbp : pair.second.sbp_parallel()) { CollectUniverse(sbp); }
  }
}
// Collect all the possible Sbp Parallel from a SbpNode
void BoxingCollector::CollectUniverse(const auto_parallel::SbpNode<cfg::NdSbpSignature>* sbp_node) {
  for (const auto& nd_sbp_sig : sbp_node->SbpSignatureObjList) { CollectUniverse(nd_sbp_sig); }
}
// Collect Sbp Parallel
void BoxingCollector::CollectUniverse(const cfg::SbpParallel& sbp) {
  if (SbpParallelUniverse.find(sbp) == SbpParallelUniverse.end()) {
    int32_t curr_size = SbpParallelUniverse.size();
    SbpParallelUniverse[sbp] = curr_size;
    id2SbpParallel.push_back(sbp);
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

// Generate the transfer rule for different combinations and hierarchies
Maybe<void> BoxingCollector::GenerateCombination(int32_t max_middle_node_num) {
  // 1D sbp does not support S->P. But it seems that we do not need to deal with it for now.
  // And we do not have 3D sbp or higher dimension.
  int32_t hierarchy_num = 2;

  // Generate possible nd_sbp lists
  cfg::NdSbp nd_sbp;
  for (int32_t dim_sbp = 0; dim_sbp < hierarchy_num; dim_sbp++) { nd_sbp.add_sbp_parallel(); }
  DfsSetNdSbp(id2SbpParallel, 0, hierarchy_num, nd_sbp, nd_sbp_lists, NdSbpUniverse);
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

  auto NotMiddleNode = [&](int32_t i, int32_t j, int32_t k, int32_t middle_node_num_ik) -> bool {
    // Not allow i -> i -> j or i -> j -> j.
    if (k == j || k == i) { return true; }
    // We add middle nodes one by one
    // Thus, we allow multiple nodes from i to k but we only accept 1 step from k to j.
    // i -> ? -> k -> j
    if (middle_nodes[k][j].size() > 0) { return true; }
    // To avoid multiple counting and bugs, the number of middle nodes between i and k
    // must be exactly middle_node_num_ik, which is (middle_node_num - 1)
    if (middle_node_num_ik) {
      if (middle_nodes[i][k].size() == 0 || middle_nodes[i][k][0].size() != middle_node_num_ik) {
        return true;
      }
    } else {
      if (middle_nodes[i][k].size() > 0) { return true; }
    }
    return false;
  };

  for (int32_t middle_node_num = 1; middle_node_num <= max_middle_node_num; middle_node_num++) {
    int32_t middle_node_num_ik = middle_node_num - 1;

    for (int32_t i = 0; i < n; i++) {
      for (int32_t j = 0; j < n; j++) {
        if (minimum_copy_cost[i][j] < cut_cost) { continue; }
        // Compute the smallest transfer cost
        // k is the middle node, i -> k -> j
        for (int32_t k = 0; k < n; k++) {
          if (NotMiddleNode(i, j, k, middle_node_num_ik)) { continue; }
          double curr_copy_cost = minimum_copy_cost[i][k] + minimum_copy_cost[k][j];
          if (curr_copy_cost < minimum_copy_cost[i][j]) {
            minimum_copy_cost[i][j] = curr_copy_cost;
          }
        }
        // If the minimum copy cost remians infinity, adding one middle node does not make it.
        if (minimum_copy_cost[i][j] > cut_cost) { continue; }
        // Find those middle nodes
        for (int32_t k = 0; k < n; k++) {
          if (NotMiddleNode(i, j, k, middle_node_num_ik)) { continue; }
          // Now we start to judge if the edge have a minimum cost
          if (minimum_copy_cost[i][k] + minimum_copy_cost[k][j]
              < minimum_copy_cost[i][j] * 1.0000001) {
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

// Ask if the boxing algorithm accepts the current sbp combination
Maybe<void> BoxingCollector::AskSbpCombination(const cfg::NdSbp& sbp_producer,
                                               const cfg::NdSbp& sbp_consumer,
                                               const BlobDesc& logical_blob_desc,
                                               const ParallelDesc& producer_parallel_desc,
                                               const ParallelDesc& consumer_parallel_desc,
                                               bool customized,
                                               std::vector<cfg::NdSbp>& middle_sbps) {
  // Check the devices and hierarchy
  // At this moment, we do not support [2, 3] -> [3, 2]
  // TODO: support [2, 3] -> [3, 2]
  CHECK_OR_RETURN(producer_parallel_desc.EqualsIgnoringDeviceType(consumer_parallel_desc))
      << "Boxing does not support transfer for different machines or devices or hierarchy";
  middle_sbps.clear();
  const auto& parallel_hierarchy = producer_parallel_desc.hierarchy();
  // Dealing with 1D sbp
  if (parallel_hierarchy->NumAxes() == 1) {
    CHECK_OR_RETURN(
        JUST(auto_parallel::ComputCopyCostBetweenTwoSbpParallel(
            sbp_producer.sbp_parallel(0), sbp_consumer.sbp_parallel(0), logical_blob_desc,
            producer_parallel_desc, consumer_parallel_desc, false, true))
        < cut_cost)
        << "Boxing does not support " << NdSbpParallelToString(sbp_producer) << " -> "
        << NdSbpParallelToString(sbp_consumer) << " for 1D sbp";
    return Maybe<void>::Ok();
  }
  // Dealing with nD sbp, n>2
  CHECK_OR_RETURN(parallel_hierarchy->NumAxes() == 2)
      << "Boxing does not support a hierarchy with dimension greater than 2";
  // Dealing with 2D sbp
  const auto& it_producer = NdSbpUniverse.find(sbp_producer);
  const auto& it_consumer = NdSbpUniverse.find(sbp_consumer);
  if (it_producer != NdSbpUniverse.end() && it_consumer != NdSbpUniverse.end()) {
    int32_t i = it_producer->second;
    int32_t j = it_consumer->second;
    // Such combination can not be support with limited middle nodes
    CHECK(minimum_copy_cost[i][j] < cut_cost)
        << "Boxing does not support " << NdSbpParallelToString(sbp_producer) << " -> "
        << NdSbpParallelToString(sbp_consumer) << " for 2D sbp";
    // Current design can deal with such combination. Do not need to insert middle nodes
    if (middle_nodes[i][j].size() == 0) { return Maybe<void>::Ok(); }
    // Find a list of middle nodes with minimum storage
    int32_t min_k = -1;
    double min_cost = cut_cost;
    for (int32_t k = 0; k < middle_nodes[i][j].size(); k++) {
      double curr_cost = 0.0;
      for (int32_t middle_sbp_id : middle_nodes[i][j][k]) {
        Shape logical_shape = logical_blob_desc.shape();
        // Storage4NdSbp would modify logical_shape2 as well
        curr_cost += auto_parallel::Storage4NdSbp(nd_sbp_lists[middle_sbp_id], logical_shape,
                                                  parallel_hierarchy);
        if (curr_cost > cut_cost) { break; }
      }
      // store k if renew minimum cost
      if (curr_cost < min_cost) {
        min_k = k;
        min_cost = curr_cost;
      }
    }

    // If we found a list of middle nodes with current boxing collector
    if (min_k >= 0) {
      for (int32_t middle_sbp_id : middle_nodes[i][j][min_k]) {
        middle_sbps.push_back(nd_sbp_lists[middle_sbp_id]);
      }
      return Maybe<void>::Ok();
    }
  }

  // // If we can not found a list of middle nodes even after customized boxing collector
  CHECK_OR_RETURN(customized) << "Boxing does not support " << NdSbpParallelToString(sbp_producer)
                              << " -> " << NdSbpParallelToString(sbp_consumer)
                              << " for Shape: " << logical_blob_desc.shape();

  // Customized boxing collector and try the algorithm again
  BoxingCollector customized_boxing_collector;
  customized_boxing_collector.CollectUniverse(logical_blob_desc.shape().NumAxes());
  customized_boxing_collector.GenerateCombination(5);
  // test debug
  customized_boxing_collector.PrintBoxingTables();
  JUST(customized_boxing_collector.AskSbpCombination(sbp_producer, sbp_consumer, logical_blob_desc,
                                                     producer_parallel_desc, consumer_parallel_desc,
                                                     false, middle_sbps));
  return Maybe<void>::Ok();
}

// Filter nd sbp from nd_sbp_lists with given logical shape
Maybe<void> BoxingCollector::FilterNdSbpList4LogicalShape(
    const BlobDesc& logical_blob_desc, const std::shared_ptr<Shape>& parallel_hierarchy) {
  for (int32_t middle_sbp_id = nd_sbp_lists.size() - 1; middle_sbp_id >= 0; middle_sbp_id--) {
    Shape logical_shape = logical_blob_desc.shape();
    if (JUST(FilterNdSbpByLogicalShape(nd_sbp_lists[middle_sbp_id], logical_shape,
                                       parallel_hierarchy))) {
      // Change the value before erasing
      // This might be true: nd_sbp_lists.size() - 1 == middle_sbp_id
      NdSbpUniverse[nd_sbp_lists[nd_sbp_lists.size() - 1]] = middle_sbp_id;
      NdSbpUniverse.erase(nd_sbp_lists[middle_sbp_id]);
      nd_sbp_lists[middle_sbp_id] = nd_sbp_lists[nd_sbp_lists.size() - 1];
      nd_sbp_lists.pop_back();
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow