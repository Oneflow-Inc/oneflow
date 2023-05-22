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

#include "oneflow/core/auto_parallel/straighten_memory_op_graph.h"
#include "oneflow/core/auto_parallel/auto_memory.h"

namespace oneflow {
namespace auto_parallel {

namespace {

void InitInOutTopoStructs(HashMap<const OpNode*, MemoryTopoStruct>& op_node2topo_struct) {
  // Generate the map from operator names to topological structure
  HashMap<std::string, MemoryTopoStruct*> op_name2topo_structs;
  for (auto& pair : op_node2topo_struct) {
    op_name2topo_structs[pair.first->op().op_name()] = &pair.second;
  }

  // Traverse the topological structures
  for (auto& pair : op_node2topo_struct) {
    auto& node = pair.first;
    auto* this_topo_struct = &pair.second;
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

void InitAllParameters(HashMap<const OpNode*, MemoryTopoStruct>& op_node2topo_struct,
                       HashMap<LogicalBlobId, int32_t>* lbi2id,
                       std::vector<MemoryTopoStruct*>* id2producer_topo_struct,
                       std::vector<std::vector<MemoryTopoStruct*>>* id2consumer_topo_structs,
                       std::vector<int64_t>* id2blob_size) {
  for (auto& pair : op_node2topo_struct) {
    const auto& producer = pair.first->op();
    auto* topo_struct = &pair.second;

    // Find all the blobs produced by this operator
    for (const auto& obn : producer.output_bns()) {
      const LogicalBlobId& lbi = producer.BnInOp2Lbi(obn);
      auto it = lbi2id->find(lbi);
      // We check existence in case of inplace operators, whose producer and consumer produce the
      // same blob
      if (it == lbi2id->end()) {
        (*lbi2id)[lbi] = id2blob_size->size();
        const BlobDesc& logical_blob_desc = pair.first->LogicalBlobDesc4Lbi(lbi);
        id2blob_size->push_back(TotalByteSize4BlobDesc(logical_blob_desc));
        id2producer_topo_struct->push_back(topo_struct);
      }
    }
  }

  // initialize the id2consumer_topo_structs
  id2consumer_topo_structs->resize(id2blob_size->size());
  // Find all the blobs consumed by this operator
  for (auto& pair : op_node2topo_struct) {
    const auto& consumer = pair.first->op();
    for (const auto& ibn : consumer.input_bns()) {
      const LogicalBlobId& lbi = consumer.BnInOp2Lbi(ibn);
      id2consumer_topo_structs->at(lbi2id->find(lbi)->second).push_back(&pair.second);
    }
  }

  // Construct all the data edges and control edges
  InitInOutTopoStructs(op_node2topo_struct);
}

}  // namespace

// Straighten a subset of the op graph
void StraightenMemorySubGraph(const std::vector<const OpNode*>& sub_graph,
                              std::vector<const OpNode*>* ordered_op_nodes) {
  // Generate topological data structure for each op node
  HashMap<const OpNode*, MemoryTopoStruct> op_node2topo_struct;
  std::vector<MemoryTopoStruct*> topo_structs;
  std::vector<MemoryTopoStruct*> ordered_topo_structs;
  HashMap<MemoryTopoStruct*, const OpNode*> topo_struct2op_node;

  // Traverse all the nodes in the sub graph
  for (const auto* node : sub_graph) {
    op_node2topo_struct.insert({node, MemoryTopoStruct(kOriginNode)});
    auto& topo_struct = op_node2topo_struct.at(node);
    topo_struct.op_node = node;
    topo_struct.is_reusable = IsProducedRegisterReusable(node->op());
    topo_structs.push_back(&topo_struct);
    topo_struct2op_node[&topo_struct] = node;
  }

  // Construct the map from a lbi to its id, producer, consumers, blob size
  HashMap<LogicalBlobId, int32_t> lbi2id;
  std::vector<MemoryTopoStruct*> id2producer_topo_struct;
  std::vector<std::vector<MemoryTopoStruct*>> id2consumer_topo_structs;
  std::vector<int64_t> id2blob_size;

  InitAllParameters(op_node2topo_struct, &lbi2id, &id2producer_topo_struct,
                    &id2consumer_topo_structs, &id2blob_size);

  StraightenMemory(&topo_structs, &lbi2id, &id2producer_topo_struct, &id2consumer_topo_structs,
                   &id2blob_size, &ordered_topo_structs);

  for (auto& ordered_topo_struct : ordered_topo_structs) {
    ordered_op_nodes->push_back(topo_struct2op_node[ordered_topo_struct]);
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