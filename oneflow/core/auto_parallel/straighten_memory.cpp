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

#include "oneflow/core/auto_parallel/straighten_memory.h"
#include "oneflow/core/auto_parallel/algorithm_util.h"
#include "oneflow/core/auto_parallel/auto_memory.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {
namespace auto_parallel {

namespace {

class NoCleaningMarkerAccMemory {
 public:
  static int32_t marker;
  int32_t status = 0;

  bool IfMarked() const { return status == marker; }
  bool IfNotMarked() const { return status != marker; }
  void Mark() { status = marker; }
  void UnMark() { status = 0; }
};

int32_t NoCleaningMarkerAccMemory::marker = 1;

void ResetNoCleaningMarkerAccMemory() { ++NoCleaningMarkerAccMemory::marker; }

void InitNoCleaningMarkerAccMemory() { NoCleaningMarkerAccMemory::marker = 1; }

class NoCleaningMarkerDescendant {
 public:
  static int32_t marker;
  int32_t status = 0;

  bool IfMarked() const { return status == marker; }
  bool IfNotMarked() const { return status != marker; }
  void Mark() { status = marker; }
  void UnMark() { status = 0; }
};

int32_t NoCleaningMarkerDescendant::marker = 1;

void ResetNoCleaningMarkerDescendant() { ++NoCleaningMarkerDescendant::marker; }

void InitNoCleaningMarkerDescendant() { NoCleaningMarkerDescendant::marker = 1; }

class TopoStruct {
 public:
  const OpNode* op_node = nullptr;
  // Memory increment = (memory of out registers) - (memory of in registers)
  int64_t memory_increment = -1;
  int32_t min_layer = -1;
  bool is_reusable = false;
  int32_t blob_id = -1;
  // blocking means that it contains a release topological structure with degree = 1 in the current
  // unexecuted graph
  TopoStruct* blocking_topo_struct = nullptr;
  // executed means that it has been executed
  bool executed = false;
  // Accumulate memory increment of all the necessary topological structures
  int64_t accumulate_memory_increment = 0;
  // Whether visited during memory accumulating
  NoCleaningMarkerAccMemory visited_acc_memory;
  // Whether visited while finding descendants
  NoCleaningMarkerDescendant visited_descendant;
  // waiting in the map before execution
  bool waiting = false;

  std::vector<TopoStruct*> in_topo_structs;
  std::vector<TopoStruct*> out_topo_structs;

  // The topo structs to be executed in a reverse order right before this topo struct
  // For example:
  // This topo struct: A, Pre topo structs: {B, C, D}
  // This topo struct: B, Pre topo structs: {E}
  // This topo struct: D, Pre topo structs: {F, G}
  // And the graph is: H -> A -> I
  // Then the execution order is H, G, F, D, C, E, B, A, I
  std::vector<TopoStruct*> pre_topo_structs;
  // The topo structs to be executed immediately after this topo struct
  std::vector<TopoStruct*> post_topo_structs;

  explicit TopoStruct(const OpNode* op_node_);
  explicit TopoStruct(int32_t blob_id_) : blob_id(blob_id_){};

  // Compute the minimum layer of this node
  int32_t ComputeMinLayer();

  void ComputeIsReusable();

  void SetAccumulateMemoryIncrement();

  void MarkDescendantFromThis2Layer(int32_t max_layer);

 private:
  int64_t AccumulateMemoryIncrement();
  // Mark all its descendant with min_layer <= max_layer
  void MarkDescendantUp2Layer(int32_t max_layer);
};

TopoStruct::TopoStruct(const OpNode* op_node_) : op_node(op_node_) { ComputeIsReusable(); }

// Compute the minimum layer of this node
int32_t TopoStruct::ComputeMinLayer() {
  if (min_layer >= 0) { return min_layer; }
  for (auto& in_topo_struct : in_topo_structs) {
    min_layer = std::max(min_layer, in_topo_struct->ComputeMinLayer());
  }
  return ++min_layer;
}

void TopoStruct::ComputeIsReusable() { is_reusable = IsProducedRegisterReusable(op_node->op()); }

int64_t TopoStruct::AccumulateMemoryIncrement() {
  int64_t total_memory_increment = memory_increment;
  visited_acc_memory.Mark();
  for (const auto& in_topo_struct : in_topo_structs) {
    // Accumulate the non-executed topological structures only once
    if ((!in_topo_struct->executed) && in_topo_struct->visited_acc_memory.IfNotMarked()) {
      total_memory_increment += in_topo_struct->AccumulateMemoryIncrement();
    }
  }
  return total_memory_increment;
}

void TopoStruct::MarkDescendantUp2Layer(int32_t max_layer) {
  if (visited_descendant.IfMarked()) { return; }
  visited_descendant.Mark();
  if (min_layer < max_layer) {
    for (auto* out_node : out_topo_structs) { out_node->MarkDescendantUp2Layer(max_layer); }
  }
}

void TopoStruct::SetAccumulateMemoryIncrement() {
  ResetNoCleaningMarkerAccMemory();
  accumulate_memory_increment = AccumulateMemoryIncrement();
}

void TopoStruct::MarkDescendantFromThis2Layer(int32_t max_layer) {
  ResetNoCleaningMarkerDescendant();
  MarkDescendantUp2Layer(max_layer);
}

// .back() return a reference. But if the original map is destroyed in the same piece of code,
// the reference would point to [0xfffffffffffffff8], giving out an error.
TopoStruct* TakeBackFromVector(const std::vector<TopoStruct*>& v) { return v.back(); }

void ComputeLayer(std::vector<TopoStruct*>* topo_structs) {
  // Initial all the layer to -1
  for (auto& topo_struct : *topo_structs) { topo_struct->min_layer = -1; }
  // Compute the minimum layer for the whole graph
  for (auto& topo_struct : *topo_structs) { topo_struct->ComputeMinLayer(); }
}

void ConnectTwoNodes(TopoStruct* producer, TopoStruct* consumer) {
  // Check if the edge exists
  int32_t in_id = CheckIndex(consumer->in_topo_structs, producer);
  if (in_id < 0) {
    consumer->in_topo_structs.push_back(producer);
    producer->out_topo_structs.push_back(consumer);
  }
}

void InitInOutTopoStructs(std::vector<TopoStruct*>* topo_structs) {
  // Generate the map from operator names to topological structure
  HashMap<std::string, TopoStruct*> op_name2topo_structs;
  for (auto& topo_struct : *topo_structs) {
    op_name2topo_structs[topo_struct->op_node->op().op_name()] = topo_struct;
  }

  // Traverse the topological structures
  for (auto& this_topo_struct : *topo_structs) {
    auto& node = this_topo_struct->op_node;
    // Initialize input nodes for edges with data
    node->ForEachNodeOnInEdge([&](OpNode* in) {
      // Since we might be looking at a sub-graph of the operator graph.
      // We need to check if the op_node exists in the sub-graph.
      auto it = op_name2topo_structs.find(in->op().op_name());
      if (it != op_name2topo_structs.end()) { ConnectTwoNodes(it->second, this_topo_struct); }
    });
    // Initialize input nodes for control edges
    for (const auto& ctrl_in_op_name : node->op().op_conf().ctrl_in_op_name()) {
      auto it = op_name2topo_structs.find(ctrl_in_op_name);
      if (it != op_name2topo_structs.end()) { ConnectTwoNodes(it->second, this_topo_struct); }
    }
  }
}

// Compute the memory increment for all the topological structures
void ComputeAllMemoryIncrement(std::vector<TopoStruct*>& topo_structs,
                               std::vector<TopoStruct>& release_topo_structs,
                               HashMap<LogicalBlobId, int32_t>& lbi2id,
                               std::vector<TopoStruct*>& id2producer_topo_struct,
                               std::vector<std::vector<TopoStruct*>>& id2consumer_topo_structs,
                               std::vector<int64_t>& id2blob_size) {
  // Compute the memory increment for produced blobs
  for (auto& topo_struct : topo_structs) { topo_struct->memory_increment = 0; }

  for (int32_t id = 0; id < id2producer_topo_struct.size(); id++) {
    const auto& topo_struct = id2producer_topo_struct[id];
    if (topo_struct->is_reusable) { topo_struct->memory_increment += id2blob_size[id]; }
  }
  // Subtract the consumed memory
  for (int32_t id = 0; id < id2consumer_topo_structs.size(); id++) {
    if (id2producer_topo_struct[id]->is_reusable) {
      auto& consumer_topo_structs = id2consumer_topo_structs[id];
      // Release the blob in the blocking node
      if (consumer_topo_structs.size() == 1) {
        id2producer_topo_struct[id]->memory_increment -= id2blob_size[id];
        continue;
      }
      // Check whether two blobs have the same consumer_topo_structs
      auto& first_consumer_outs = consumer_topo_structs[0]->out_topo_structs;
      bool not_merged = true;
      for (int32_t out_id = first_consumer_outs.size() - 1; out_id >= 0; out_id--) {
        int32_t curr_release_blob_id = first_consumer_outs[out_id]->blob_id;
        if (curr_release_blob_id == -1) { break; }
        // Compare whether the consumer_topo_structs are the same
        const auto& curr_topo_structs = id2consumer_topo_structs[curr_release_blob_id];
        bool is_same = curr_topo_structs.size() == consumer_topo_structs.size();
        for (int32_t consumer_id = 0; consumer_id < consumer_topo_structs.size(); consumer_id++) {
          if (consumer_topo_structs[consumer_id] != curr_topo_structs[consumer_id]) {
            is_same = false;
            break;
          }
        }
        // If they have the same consumer_topo_structs, merge them
        if (is_same) {
          first_consumer_outs[out_id]->memory_increment -= id2blob_size[id];
          not_merged = false;
          break;
        }
      }
      // If they have different consumer_topo_structs, add a new release_topo_struct
      if (not_merged) {
        release_topo_structs.emplace_back(id);
        auto& release_topo_struct = release_topo_structs.back();
        topo_structs.push_back(&release_topo_struct);
        release_topo_struct.memory_increment = -id2blob_size[id];
        // We need to execute all the consumers before releasing the blob
        for (auto& consumer_topo_struct : consumer_topo_structs) {
          ConnectTwoNodes(consumer_topo_struct, &release_topo_struct);
        }
      }
    }
  }
}

void InitAllParameters(std::vector<TopoStruct*>* topo_structs,
                       std::vector<TopoStruct>* release_topo_structs,
                       HashMap<LogicalBlobId, int32_t>* lbi2id,
                       std::vector<std::vector<TopoStruct*>>* id2consumer_topo_structs,
                       std::vector<int64_t>* id2blob_size) {
  // Initialize the no cleaning marker
  InitNoCleaningMarkerAccMemory();
  InitNoCleaningMarkerDescendant();

  // Construct the map from a lbi to its id, consumers, blob size
  std::vector<TopoStruct*> id2producer_topo_struct;

  for (auto& topo_struct : *topo_structs) {
    const auto& producer = topo_struct->op_node->op();

    // Find all the blobs produced by this operator
    for (const auto& obn : producer.output_bns()) {
      const LogicalBlobId& lbi = producer.BnInOp2Lbi(obn);
      auto it = lbi2id->find(lbi);
      // We check existence in case of inplace operators, whose producer and consumer produce the
      // same blob
      if (it == lbi2id->end()) {
        (*lbi2id)[lbi] = id2blob_size->size();
        const BlobDesc& logical_blob_desc = topo_struct->op_node->LogicalBlobDesc4Lbi(lbi);
        id2blob_size->push_back(TotalByteSize4BlobDesc(logical_blob_desc));
        id2producer_topo_struct.push_back(topo_struct);
      }
    }
  }

  // Reserve the space for release topological structures
  // We do not define a copy method for TopoStruct. During each size expansion, the data might be
  // screwed up if we do not reserve the space.
  release_topo_structs->reserve(id2blob_size->size());
  // initialize the id2consumer_topo_structs
  id2consumer_topo_structs->resize(id2blob_size->size());
  // Find all the blobs consumed by this operator
  for (auto& topo_struct : *topo_structs) {
    const auto& consumer = topo_struct->op_node->op();
    for (const auto& ibn : consumer.input_bns()) {
      const LogicalBlobId& lbi = consumer.BnInOp2Lbi(ibn);
      id2consumer_topo_structs->at(lbi2id->find(lbi)->second).push_back(topo_struct);
    }
  }

  for (int32_t id = 0; id < id2consumer_topo_structs->size(); id++) {
    if (id2consumer_topo_structs->at(id).empty()) {
      // If a blob does not have a consumer, then the blob is consumed by its producer itself
      id2consumer_topo_structs->at(id).push_back(id2producer_topo_struct[id]);
    } else {
      // Sort the consumer topological structure for later matching
      std::sort(id2consumer_topo_structs->at(id).begin(), id2consumer_topo_structs->at(id).end(),
                [](const TopoStruct* a, const TopoStruct* b) { return a < b; });
    }
  }

  // Construct all the data edges and control edges
  InitInOutTopoStructs(topo_structs);

  // Compute the memory increment for all the topological structures
  ComputeAllMemoryIncrement(*topo_structs, *release_topo_structs, *lbi2id, id2producer_topo_struct,
                            *id2consumer_topo_structs, *id2blob_size);
}

void ClipOneEdge(TopoStruct* producer, TopoStruct* consumer) {
  CheckAndRemoveFrom(producer->out_topo_structs, consumer);
  CheckAndRemoveFrom(consumer->in_topo_structs, producer);
}


void EatNodes(std::vector<TopoStruct*>& topo_structs) {
  for (int32_t id = topo_structs.size() - 1; id >= 0; id--) {
    auto* node = topo_structs[id];
    if (node->memory_increment >= 0) {
      // A positive node with only one output would be executed at the last moment before the
      // execution of its output
      if (node->out_topo_structs.size() == 1) {
        // d: a, b, c -> d(+) -> g
        // g: b, d, e, f -> g -> ...
        // d has non-negative memory increment (>=0), d only have one out edge: d -> g.
        // But g might have multiple inputs.
        auto* out_node = node->out_topo_structs[0];
        // Merge d into g: (d)g
        out_node->pre_topo_structs.push_back(node);
        out_node->memory_increment += node->memory_increment;
        // Clip d -> g
        ClipOneEdge(node, out_node);
        // g takes all the inputs from d
        // g: a, b, c, e, f -> (d)g -> ...
        // Note that b is also an input of the origin g
        // and we need to make sure that b does not occur twice.
        for (auto* in_node : node->in_topo_structs) {
          // Insert a -> g, b -> g, c -> g
          ConnectTwoNodes(in_node, out_node);
          // Clip a -> d, b -> d, c -> d
          ClipOneEdge(in_node, node);
        }
        // Eliminate d
        RemoveFrom(topo_structs, id);
      }
    } else {
      // A negative node with only one input would be executed immediately after the execution of
      // its input
      if (node->in_topo_structs.size() == 1) {
        // b: a -> b(-) -> c, d, e
        // a: ... -> a -> b, d, f, g
        // b has negative memory increment (<0), b only have one in edge: a -> b.
        // But a might have multiple outputs
        auto* in_node = node->in_topo_structs[0];
        // Merge b into a: a(b)
        in_node->post_topo_structs.push_back(node);
        in_node->memory_increment += node->memory_increment;  // Clip a -> b
        ClipOneEdge(in_node, node);
        // a takes all the outputs from b
        // a: ... -> a(b) -> c, d, e, f, g
        // Note tht d is also an output of the origin a
        // and we need to make sure that d does not occur twice
        for (auto* out_node : node->out_topo_structs) {
          // Insert a -> c, a -> d, a -> e
          ConnectTwoNodes(in_node, out_node);
          // Clip b -> c, b -> d, b -> e
          ClipOneEdge(node, out_node);
        }
        // Eliminate b
        RemoveFrom(topo_structs, id);
      }
    }
  }
}

// Clip those useless edges
// For example: a -> b -> c -> d, a -> c, a -> d.
// Then we could clip the two edges a -> c and a -> d.
void ClipEdges(std::vector<TopoStruct*>& topo_structs) {
  // In the implementation, we only focus on a node with multiple inputs,
  // since max(in_nodes->min_layer) == this_node->min_layer - 1.
  for (auto* node : topo_structs) {
    // Suppose we have multiple input nodes
    // a, b, c -> d
    if (node->in_topo_structs.size() >= 2) {
      int32_t max_layer = node->min_layer - 1;
      for (int32_t in_id = node->in_topo_structs.size() - 1; in_id >= 0; in_id--) {
        auto* in_node = node->in_topo_structs[in_id];
        // Find all the descendants of node a
        in_node->MarkDescendantFromThis2Layer(max_layer);
        for (auto* brother : node->in_topo_structs) {
          // If we found a -> ... -> b (or a -> ... -> c)
          // The first judgement is to make sure we are comparing a different node,
          // i.e., brother != in_node
          if (brother->min_layer > in_node->min_layer && brother->visited_descendant.IfMarked()) {
            // Remove a -> d
            // Be careful that we need to remove the edge from two sides.
            ClipOneEdge(in_node, node);
            break;
          }
        }
      }
    }
  }
}

void GraphSimplification(std::vector<TopoStruct*>& topo_structs) {
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "Origin, Topo size: " << topo_structs.size() << std::endl;
  }
  EatNodes(topo_structs);
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "After 1st Eats, Topo size: " << topo_structs.size() << std::endl;
  }
  ComputeLayer(&topo_structs);
  ClipEdges(topo_structs);
  EatNodes(topo_structs);
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "2nd, clip and eat, Topo size: " << topo_structs.size() << std::endl;
  }
  ComputeLayer(&topo_structs);
  ClipEdges(topo_structs);
  EatNodes(topo_structs);
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "3rd, clip and eat, Topo size: " << topo_structs.size() << std::endl;
  }
  // TODO: Clip edges
}

void StraightenMemoryOpNodes(HashMap<const OpNode*, TopoStruct>& op_node2topo_struct,
                             std::vector<TopoStruct*>* topo_structs,
                             HashMap<LogicalBlobId, int32_t>* lbi2id,
                             std::vector<std::vector<TopoStruct*>>* id2consumer_topo_structs,
                             std::vector<int64_t>* id2blob_size,
                             std::vector<TopoStruct*>* ordered_topo_structs) {
  // The number of executing topographical structures
  int32_t executing_topo_struct_num = topo_structs->size();
  // Extra release topological structures
  std::vector<TopoStruct> release_topo_structs;
  InitAllParameters(topo_structs, &release_topo_structs, lbi2id, id2consumer_topo_structs,
                    id2blob_size);

  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "Print all the topological structures:" << std::endl;
    int64_t total_memory = 0;
    int32_t total_in_size = 0, total_out_size = 0;
    for (const auto& topo_struct : *topo_structs) {
      topo_struct->SetAccumulateMemoryIncrement();
      std::cout << "In size: " << topo_struct->in_topo_structs.size()
                << ", out size: " << topo_struct->out_topo_structs.size() << ", "
                << ", accumulate memory increment: " << topo_struct->accumulate_memory_increment
                << ", ";
      total_in_size += topo_struct->in_topo_structs.size();
      total_out_size += topo_struct->out_topo_structs.size();
      if (topo_struct->blob_id != -1) {
        std::cout << "Blob id: " << topo_struct->blob_id
                  << " Memory increment: " << topo_struct->memory_increment << std::endl;
      } else {
        std::cout << "Op node: " << topo_struct->op_node->op().op_name()
                  << " Memory increment: " << topo_struct->memory_increment << std::endl;
      }
      total_memory += topo_struct->memory_increment;
    }
    std::cout << "Total memory: " << total_memory << ", total in size: " << total_in_size
              << ", total out size: " << total_out_size << std::endl;
  }

  GraphSimplification(*topo_structs);

  // Those nodes that we need to visit their descendants
  // At the beginning, them would be the source nodes.
  // After each execution, them would be those executed nodes.
  std::vector<TopoStruct*> prepare_topo_structs;
  // wait in the map
  std::map<int64_t, std::vector<TopoStruct*>> waiting_map;
  // Erase a node from the waiting map
  auto StopWaiting = [&](TopoStruct* node) {
    if (node->waiting) {
      node->waiting = false;
      auto& waiting_list = waiting_map[node->accumulate_memory_increment];
      if (waiting_list.size() == 1) {
        waiting_map.erase(node->accumulate_memory_increment);
      } else {
        // Erase node from the waiting list
        for (int32_t i = waiting_list.size() - 1; i >= 0; i--) {
          if (waiting_list[i] == node) {
            waiting_list[i] = waiting_list[waiting_list.size() - 1];
            waiting_list.pop_back();
            break;
          }
        }
      }
    }
  };
  // Wait in the map
  auto Wait = [&](TopoStruct* node) {
    if (node->executed) { return; }
    StopWaiting(node);
    node->SetAccumulateMemoryIncrement();
    waiting_map[node->accumulate_memory_increment].push_back(node);
    node->waiting = true;
  };
  // Visit one node
  std::function<void(TopoStruct*)> Visit = [&](TopoStruct* node) {
    if (node->visited_descendant.IfMarked()) { return; }
    node->visited_descendant.Mark();
    if (node->memory_increment >= 0 || node->executed) {
      for (auto* out_node : node->out_topo_structs) { Visit(out_node); }
    } else {
      Wait(node);
    }
  };
  // Prepare all the release nodes before picking one for the next round
  auto Prepare = [&]() {
    ResetNoCleaningMarkerDescendant();
    for (auto* node : prepare_topo_structs) { Visit(node); }
  };
  int64_t peak_memory = 0;
  int64_t total_memory = 0;
  std::function<void(TopoStruct*)> ExecuteOpNode = [&](TopoStruct* node) {
    for (int32_t i = node->pre_topo_structs.size() - 1; i >= 0; i--) {
      ExecuteOpNode(node->pre_topo_structs[i]);
    }
    if (node->op_node) { ordered_topo_structs->push_back(node); }
    for (auto* post_node : node->post_topo_structs) { ExecuteOpNode(post_node); }
  };
  // Execute one node and its ancestors
  std::function<void(TopoStruct*)> Execute = [&](TopoStruct* node) {
    // Post-order traversal
    for (auto* in_node : node->in_topo_structs) {
      if (!in_node->executed) { Execute(in_node); }
    }
    // Execute the current node
    ExecuteOpNode(node);
    node->executed = true;
    total_memory += node->memory_increment;
    if (total_memory > peak_memory) { peak_memory = total_memory; }
    StopWaiting(node);
    prepare_topo_structs.push_back(node);
    if (GlobalProcessCtx::Rank() == 0) {
      std::cout << "Executing ";
      if (node->op_node) {
        std::cout << node->op_node->op().op_name();
      } else {
        std::cout << "blob id: " << node->blob_id;
      }
      std::cout << ", memory increment: " << node->memory_increment
                << ", current total: " << total_memory << std::endl;
    }
  };

  // TODO: expand to all the topo structures
  // Initialize source topological structures
  for (int32_t i = 0; i < topo_structs->size(); i++) {
    if (topo_structs->at(i)->in_topo_structs.empty()) {
      prepare_topo_structs.push_back(topo_structs->at(i));
    }
  }
  // Straighten memory
  while (ordered_topo_structs->size() < executing_topo_struct_num) {
    // Prepare the release node for this round
    Prepare();
    // Clean up the prepare_topo_structs before executing any node
    prepare_topo_structs.clear();
    // Pick the one with the smallest accumulate memory increment and then execute it
    if (waiting_map.empty()) { break; }
    Execute(TakeBackFromVector(waiting_map.begin()->second));
  }

  // Execute the rest of the nodes
  for (auto& node : *topo_structs) {
    if (!node->executed) {
      CHECK(node->memory_increment >= 0)
          << "All the blobs should be release during straighten memory!";
      Execute(node);
    }
  }
}
}  // namespace

// Straighten a subset of the op graph
void StraightenMemorySubGraph(const std::vector<const OpNode*>& sub_graph,
                              std::vector<const OpNode*>* ordered_op_nodes) {
  // Generate topological data structure for each op node
  HashMap<const OpNode*, TopoStruct> op_node2topo_struct;
  std::vector<TopoStruct*> topo_structs;
  std::vector<TopoStruct*> ordered_topo_structs;

  // Traverse all the nodes in the sub graph
  for (const auto& node : sub_graph) {
    op_node2topo_struct.insert({node, TopoStruct(node)});
    topo_structs.push_back(&op_node2topo_struct.at(node));
  }

  // Construct the map from a lbi to its id, consumers, blob size
  HashMap<LogicalBlobId, int32_t> lbi2id;
  std::vector<std::vector<TopoStruct*>> id2consumer_topo_structs;
  std::vector<int64_t> id2blob_size;

  StraightenMemoryOpNodes(op_node2topo_struct, &topo_structs, &lbi2id, &id2consumer_topo_structs,
                          &id2blob_size, &ordered_topo_structs);

  for (auto& ordered_topo_struct : ordered_topo_structs) {
    ordered_op_nodes->push_back(ordered_topo_struct->op_node);
  }
}

// Straighten the whole op graph
void StraightenMemoryOpGraph(const OpGraph& op_graph,
                             std::vector<const OpNode*>* ordered_op_nodes) {
  std::vector<const OpNode*> sub_graph;

  // Traverse and store all the nodes in the op graph
  op_graph.ForEachNode([&](OpNode* node) { sub_graph.push_back(node); });

  StraightenMemorySubGraph(sub_graph, ordered_op_nodes);
}

}  // namespace auto_parallel
}  // namespace oneflow