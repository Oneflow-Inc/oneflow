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

#include "oneflow/core/graph/straighten_memory_task_graph.h"
#include "oneflow/core/auto_parallel/straighten_memory.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

namespace {

std::string GenerateKey4TaskNode(TaskNode* node,
                                 HashMap<const TaskNode*, std::string>& task_node2keys) {
  auto it = task_node2keys.find(node);
  if (it == task_node2keys.end()) {
    // Generate the key to determine the same task nodes
    if (dynamic_cast<TransportTaskNode*>(node)) {
      // Deal with the communication
      std::string key =
          dynamic_cast<TransportTaskNode*>(node)->lbi().ShortDebugString() + ", producer: ";
      node->ForEachNodeOnInEdge(
          [&](TaskNode* in) { key += GenerateKey4TaskNode(in, task_node2keys); });
      task_node2keys[node] = key;
    } else if (node->GetTaskType() == TaskType::kNormalForward) {
      // Deal with the normal computation nodes
      task_node2keys[node] = dynamic_cast<CompTaskNode*>(node)->op()->op_name();
    } else {
      // Tick and some other nodes
      task_node2keys[node] = node->VisualStr();
    }
    it = task_node2keys.find(node);
  }
  // Return the stored key
  return it->second;
}

void InitInOutTopoStructs(HashMap<const TaskNode*, std::string>& task_node2keys,
                          HashMap<std::string, std::vector<TaskNode*>>& key2task_nodes,
                          HashMap<std::string, auto_parallel::MemoryTopoStruct>& key2topo_struct) {
  for (auto& pair : task_node2keys) {
    auto* consumer_topo_struct = &key2topo_struct.at(pair.second);
    pair.first->ForEachNodeOnInEdge([&](TaskNode* in) {
      // Since we might be looking at a sub-graph of the task graph.
      // We need to check if the node exists in the sub-graph. (separation compilation)
      auto it_task_node2key = task_node2keys.find(in);
      if (it_task_node2key == task_node2keys.end()) { return; }
      auto it_key2topo_struct = key2topo_struct.find(it_task_node2key->second);
      if (it_key2topo_struct == key2topo_struct.end()) { return; }
      auto_parallel::ConnectTwoNodes(&it_key2topo_struct->second, consumer_topo_struct);
    });
  }
}

void InitAllParameters(
    HashMap<const TaskNode*, std::string>& task_node2keys,
    HashMap<std::string, std::vector<TaskNode*>>& key2task_nodes,
    HashMap<std::string, auto_parallel::MemoryTopoStruct>& key2topo_struct,
    HashMap<RegstDesc*, int32_t>* register2id,
    std::vector<auto_parallel::MemoryTopoStruct*>* id2producer_topo_struct,
    std::vector<std::vector<auto_parallel::MemoryTopoStruct*>>* id2consumer_topo_structs,
    std::vector<int64_t>* id2blob_size, std::vector<bool>* id2is_reusable) {
  for (auto& pair : key2topo_struct) {
    const auto& topo_struct = &pair.second;

    // Find all the registers produced by these task nodes
    for (auto* task_node : key2task_nodes.at(pair.first)) {
      for (const auto& out_blob7produced_register : task_node->produced_regsts()) {
        auto* produced_register = out_blob7produced_register.second.get();
        auto it = register2id->find(produced_register);
        if (it == register2id->end()) {
          (*register2id)[produced_register] = id2blob_size->size();
          RegstDescProto temp_proto;
          produced_register->ToProto(&temp_proto);
          id2blob_size->push_back(RtRegstDesc(temp_proto).TotalMainByteSize4AllRegst());
          id2producer_topo_struct->push_back(topo_struct);
          id2is_reusable->push_back(produced_register->enable_reuse_mem());
          id2consumer_topo_structs->emplace_back();
          auto& consumer_topo_structs = id2consumer_topo_structs->back();
          consumer_topo_structs.reserve(produced_register->consumers().size());
          for (auto* consumer : produced_register->consumers()) {
            auto_parallel::CheckAndInsert(consumer_topo_structs,
                                          &(key2topo_struct.at(task_node2keys.at(consumer))));
          }
        }
      }
    }
  }

  // Construct all the data edges and control edges
  InitInOutTopoStructs(task_node2keys, key2task_nodes, key2topo_struct);
}

}  // namespace

// Straighten the task graph to reduce memory
void StraightenMemoryTaskGraph(TaskGraph* task_graph, std::vector<TaskNode*>* ordered_task_nodes) {
  // Link the task nodes with key
  HashMap<const TaskNode*, std::string> task_node2keys;
  HashMap<std::string, std::vector<TaskNode*>> key2task_nodes;
  task_graph->ForEachNode([&](TaskNode* node) {
    key2task_nodes[GenerateKey4TaskNode(node, task_node2keys)].push_back(node);
  });
  // Generate topological data structure for each task node
  HashMap<std::string, auto_parallel::MemoryTopoStruct> key2topo_struct;
  HashMap<auto_parallel::MemoryTopoStruct*, std::string> topo_struct2key;
  std::vector<auto_parallel::MemoryTopoStruct*> topo_structs;
  std::vector<auto_parallel::MemoryTopoStruct*> ordered_topo_structs;

  // Traverse all the keys in the task graph
  for (auto& pair : key2task_nodes) {
    key2topo_struct.insert(
        {pair.first, auto_parallel::MemoryTopoStruct(auto_parallel::kOriginNode)});
    auto& topo_struct = key2topo_struct.at(pair.first);
    topo_structs.push_back(&topo_struct);
    topo_struct2key[&topo_struct] = pair.first;
  }

  // Construct the map from a register to its id, producer, consumers, blob size
  HashMap<RegstDesc*, int32_t> register2id;
  std::vector<auto_parallel::MemoryTopoStruct*> id2producer_topo_struct;
  std::vector<std::vector<auto_parallel::MemoryTopoStruct*>> id2consumer_topo_structs;
  std::vector<int64_t> id2blob_size;
  std::vector<bool> id2is_reusable;

  InitAllParameters(task_node2keys, key2task_nodes, key2topo_struct, &register2id,
                    &id2producer_topo_struct, &id2consumer_topo_structs, &id2blob_size,
                    &id2is_reusable);

  auto_parallel::StraightenMemory(&topo_structs, id2producer_topo_struct, &id2consumer_topo_structs,
                                  id2blob_size, id2is_reusable, &ordered_topo_structs);

  for (auto& ordered_topo_struct : ordered_topo_structs) {
    for (auto* task_node : key2task_nodes.at(topo_struct2key.at(ordered_topo_struct))) {
      ordered_task_nodes->push_back(task_node);
    }
  }
}

}  // namespace oneflow
