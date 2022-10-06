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
#include "oneflow/core/graph/straighten_nodes.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/task.pb.h"

namespace oneflow {

namespace {

enum TaskClassifier : int {
  kWaitingTransfer = 0,
  kWaitingComputation = 1,
  kRunASAP = 2,
  kRunALAP = 3
};

class TopoStruct {
 public:
  TaskNode* node = nullptr;
  int32_t min_layer = -1;
  int32_t tributary_layer = -1;
  bool on_mainstem = false;
  int32_t counter = 0;
  int32_t min_distance2transfer = -1;
  TopoStruct* next_same_node = nullptr;
  // We can have some other nodes in it for example
  // SbpNode<NdSbpSignature>* node;
  // SbpEdge<NdSbpSignature>* node;
  // Or we can omit all the pointers and leave all the useful parameters.

  // Drop down the tributary layer
  void DropTributaryLayer(int32_t upper_bound);

  void SpreadTributaryLayer(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct);

  void SpreadMainstem(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct);

  // The minimum computation distance from the beginning of this op to the next transfer
  int32_t GetMinDistance2Transfer(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct);

  // deciding parameter
  // i = 0: those with small tributary layers go first
  // i = 1: those with small minimum distance to transfer go first
  // i = 2: first in first out
  // i = 3: those with large tributary layers go first
  // i = 4: those with long distance to transfer go first
  // i = 5: last in first out
  int32_t GetDecidingParameter(int32_t i) const;
};

// move the head from source to target
void MoveFrontBetweenMaps(std::map<int32_t, TopoStruct*>& source,
                          std::map<int32_t, TopoStruct*>& target) {
  if (!source.empty()) {
    const auto& front = source.begin();
    target[front->first] = front->second;
    source.erase(front);
  }
};

bool ShouldRunASAP(TaskType task_type) {
  // They are sorted according to frequency of occurrences
  switch (task_type) {
    // We mark the number of occurrences in bert
    case TaskType::kDeviceTick:                  // 38
    case TaskType::kTick:                        // 8
    case TaskType::kSrcSubsetTick:               // 6
    case TaskType::kDstSubsetTick:               // 6
    case TaskType::kCriticalSectionWaitTick:     // 4
    case TaskType::kWaitAndSendIds:              // 2
    case TaskType::kPack:                        // 0
    case TaskType::kUnpack:                      // 0
    case TaskType::kRepeat:                      // 0
    case TaskType::kAcc:                         // 0
    case TaskType::kSourceTick:                  // 0
    case TaskType::kAccTick:                     // 0
    case TaskType::kCase:                        // 0
    case TaskType::kEsac:                        // 0
    case TaskType::kReentrantLock: return true;  // 0
    default: return false;
  }
}

bool IsTransferNode(TaskType task_type) {
  // return task_type == 12 || task_type == 13 || (48 <= task_type && task_type <= 64);
  // They are sorted according to frequency of occurrences
  switch (task_type) {
    // We mark the number of occurrences in bert
    case TaskType::kCollectiveBoxingGeneric:        // 76
    case TaskType::kNcclSendRecvBoxing:             // ?
    case TaskType::kCopyHd:                         // 27
    case TaskType::kSliceBoxing:                    // 16
    case TaskType::kCopyCommNet:                    // 12
    case TaskType::kCollectiveBoxingPack:           // 8
    case TaskType::kCollectiveBoxingUnpack:         // 8
    case TaskType::kBoxingZeros:                    // 3
    case TaskType::kDistributeConcat:               // 0
    case TaskType::kDistributeSplit:                // 0
    case TaskType::kBoxingIdentity:                 // 0
    case TaskType::kDecodeH2D:                      // 0
    case TaskType::kSspVariableProxy: return true;  // 0
    default: return false;
  }
}

// Classifier for the set according to the task type
TaskClassifier GetTaskClassifier(const TaskNode* node) {
  // Check task.pb.h for detail
  // They are sorted according to frequency of judgement
  // frequency of judgement = the number of occurrences / the times of judgement
  TaskType task_type = node->GetTaskType();
  if (task_type == TaskType::kNormalForward) { return TaskClassifier::kWaitingComputation; }
  if (IsTransferNode(task_type)) { return TaskClassifier::kWaitingTransfer; }
  if (task_type == TaskType::kCallbackNotify) { return TaskClassifier::kRunALAP; }
  if (ShouldRunASAP(task_type)) { return TaskClassifier::kRunASAP; }
  CHECK(false) << "Unclassified or invalid task type (" << task_type << ") showing up";
  // Throw a kRunASAP which means ignoring this node in the algorithm
  return TaskClassifier::kRunASAP;
}

// Drop down the maximum layer with the minimum layer form consumer
void TopoStruct::DropTributaryLayer(int32_t upper_bound) {
  if (upper_bound < tributary_layer || tributary_layer < 0) { tributary_layer = upper_bound; }
}

// Should initialize the counter to be the number of out edges
// Compute maximum layer for tributaries
void TopoStruct::SpreadTributaryLayer(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct) {
  if (counter || min_layer <= 0) { return; }
  int32_t producer_max_lay = 0;
  if (on_mainstem) {
    producer_max_lay = min_layer - 1;
  } else {
    // On a tributary, the operator could be run later.
    producer_max_lay = tributary_layer;
  }
  node->ForEachNodeOnInEdge([&](TaskNode* in) {
    auto& topo_struct_in = task_node2topo_struct->at(in);
    topo_struct_in.DropTributaryLayer(producer_max_lay);
    --topo_struct_in.counter;
    if (topo_struct_in.counter == 0) { topo_struct_in.SpreadTributaryLayer(task_node2topo_struct); }
  });
  // Reduce counter to -1 to avoid visiting again
  counter--;
}

// Judge if this node is on the mainstem
// If so, judge it for its producer/upstream nodes
void TopoStruct::SpreadMainstem(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct) {
  // Skip it if this node is already judged.
  if (on_mainstem) { return; }
  CHECK_GE(min_layer, 0) << "TopoStruct not initialized!";
  on_mainstem = true;
  // If I am in the mainstem, then all the children with (min_layer >= my layer id - 1) would be
  // considered as in the mainstem
  node->ForEachNodeOnInEdge([&](TaskNode* in) {
    auto& topo_struct_in = task_node2topo_struct->at(in);
    if (topo_struct_in.min_layer == min_layer - 1) {
      topo_struct_in.SpreadTributaryLayer(task_node2topo_struct);
    }
  });
}

// The minimum computation distance from the beginning of this op to the next transfer
int32_t TopoStruct::GetMinDistance2Transfer(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct) {
  if (min_distance2transfer >= 0) { return min_distance2transfer; }
  // if this node is a transfer node
  if (IsTransferNode(node->GetTaskType())) {
    min_distance2transfer = 0;
    return min_distance2transfer;
  }
  // Otherwise, initialize it with a large number
  // Well, the total number in the task graph is large enough
  min_distance2transfer = task_node2topo_struct->size();
  node->ForEachNodeOnOutEdge([&](TaskNode* out) {
    min_distance2transfer =
        std::min(min_distance2transfer,
                 task_node2topo_struct->at(out).GetMinDistance2Transfer(task_node2topo_struct));
  });
  ++min_distance2transfer;
  return min_distance2transfer;
}

// deciding parameter
// i = 0: those with small tributary layers go first
// i = 1: those with small minimum distance to transfer go first
// i = 2: first in first out
// i = 3: those with large tributary layers go first
// i = 4: those with long distance to transfer go first
// i = 5: last in first out
int32_t TopoStruct::GetDecidingParameter(int32_t i) const {
  int32_t sign = 1;
  if (i >= 3) {
    i -= 3;
    sign = -1;
  }
  switch (i) {
    case 0: return sign * tributary_layer;
    case 1: return sign * min_distance2transfer;
    case 2: return sign * min_layer;
  }
  return 0;
}

// Find the mainstem of the task graph, then reduce the wait time for tributaries
void FindMainstem(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct) {
  // Find the maximum layer number
  int32_t max_min_layer = -1;
  for (const auto& pair : *task_node2topo_struct) {
    if (max_min_layer < pair.second.min_layer) { max_min_layer = pair.second.min_layer; }
  }
  // All the nodes with min_layer>=mainstem_end_id would be considered as mainstem nodes
  // The last 5 layers would be considered as in mainstem anyway.
  int32_t mainstem_end_id = max_min_layer - 4;
  for (auto& pair : *task_node2topo_struct) {
    auto& topo_struct = pair.second;
    // Initialize the counter and Tributary Layer
    topo_struct.counter = pair.first->out_edges().size();
    topo_struct.tributary_layer = max_min_layer;
    // Find out all the nodes on the mainstem.
    if (topo_struct.min_layer >= mainstem_end_id) {
      topo_struct.SpreadMainstem(task_node2topo_struct);
    }
  }

  for (auto& pair : *task_node2topo_struct) {
    // Compute maximum layer for tributaries
    pair.second.SpreadTributaryLayer(task_node2topo_struct);
    // Set the min_distance2transfer for each topological structure
    pair.second.GetMinDistance2Transfer(task_node2topo_struct);
  }
}

}  // anonymous namespace

void StraightenNodes(TaskGraph* task_graph, std::vector<TaskNode*>* ordered_task_nodes) {
  // The function for settle the order in the graph
  int64_t order_in_graph = 0;

  // Generate topological data structure for each task node
  HashMap<TaskNode*, TopoStruct> task_node2topo_struct;
  // Determine the same nodes which should run simultaneously
  HashMap<int32_t, HashMap<int32_t, std::map<int32_t, TopoStruct*>>>
      task_type2machine_id2node_id2topo_structs;
  std::map<int32_t, TopoStruct*> min_node_id2topo_struct;
  int32_t previous_min_layer = 0;
  task_graph->TopoForEachNode([&](TaskNode* node) {
    auto& topo_struct = task_node2topo_struct[node];
    topo_struct.node = node;
    if (node->in_edges().empty()) {
      topo_struct.min_layer = 0;
    } else {
      int32_t max_min_layer = 0;
      node->ForEachNodeOnInEdge([&](TaskNode* in) {
        max_min_layer = std::max(max_min_layer, task_node2topo_struct[in].min_layer);
      });
      topo_struct.min_layer = max_min_layer + 1;
      // Deal with all the nodes with min_layer=previous_min_layer
      if (max_min_layer >= previous_min_layer) {
        // Using "7" to represent "and"
        // a7b means a pair (a, b)
        for (auto& task_type7machine_id2node_id2topo_structs :
             task_type2machine_id2node_id2topo_structs) {
          auto& machine_id2node_id2topo_structs = task_type7machine_id2node_id2topo_structs.second;
          // Initializing the smallest node id for each machine
          for (auto& machine_id7node_id2topo_structs : machine_id2node_id2topo_structs) {
            MoveFrontBetweenMaps(machine_id7node_id2topo_structs.second, min_node_id2topo_struct);
          }

          while (!min_node_id2topo_struct.empty()) {
            // auto* topo_struct_min_node_id = min_node_id2topo_struct.begin()->second;
            // Store the same nodes in different machines
            std::vector<TopoStruct*> same_nodes;
            for (auto& min_node_id7topo_struct : min_node_id2topo_struct) {
              auto* curr_topo_struct = min_node_id7topo_struct.second;
              // Find out all the same nodes
              // Stop using Visual string before we find a better key
              // Currently we can use the topological structure and node id to decide the same nodes
              same_nodes.push_back(curr_topo_struct);
            }
            // Cyclize them
            for (int32_t i = 1; i < same_nodes.size(); i++) {
              same_nodes[i - 1]->next_same_node = same_nodes[i];
            }
            (*same_nodes.rbegin())->next_same_node = same_nodes[0];
            // Delete them and add new candidates
            for (auto* same_node_topo_struct : same_nodes) {
              // Erase them from min_node_id2topo_struct
              min_node_id2topo_struct.erase(same_node_topo_struct->node->node_id());
              // Add new candidate
              MoveFrontBetweenMaps(
                  machine_id2node_id2topo_structs[same_node_topo_struct->node->machine_id()],
                  min_node_id2topo_struct);
            }
          }
        }
        // Renew the previous min_layer at the end
        previous_min_layer = topo_struct.min_layer;
      }
    }
    // Put the topo structure into the map, waiting for determine the same nodes
    task_type2machine_id2node_id2topo_structs[node->GetTaskType()][node->machine_id()]
                                             [node->node_id()] = &topo_struct;
  });

  // Generate other parameters in the topological data structure
  FindMainstem(&task_node2topo_struct);

  VLOG(3) << "Straightening order: " << 5 << ", " << 3;

  // Order in the waiting sets
  // Decide which node should run first
  struct comp {
    bool operator()(const TopoStruct* a, const TopoStruct* b) const {
      // NOTE: Leave these code for debugging in the future
      // static std::vector<int64_t> decide_parameters({ParseIntegerFromEnv("Parameter0", 5),
      //                                                ParseIntegerFromEnv("Parameter1", 3),
      //                                                ParseIntegerFromEnv("Parameter2", 5)});
      // The best parameter set is {5, 3}
      static std::vector<int64_t> decide_parameters({5, 3});
      for (int32_t decide_parameter : decide_parameters) {
        int32_t decide_parameter_a = a->GetDecidingParameter(decide_parameter);
        int32_t decide_parameter_b = b->GetDecidingParameter(decide_parameter);
        if (decide_parameter_a != decide_parameter_b) {
          return decide_parameter_a < decide_parameter_b;
        }
      }
      return a->node->node_id() < b->node->node_id();
    }
  };

  // Classify sets for the task nodes
  // std::set<TopoStruct*, comp> waiting_transfer; // 0, TaskClassifier::kWaitingTransfer
  // std::set<TopoStruct*, comp> waiting_computation; // 1, TaskClassifier::kWaitingComputation
  // std::set<TopoStruct*, comp> run_asap;  // 2, TaskClassifier::kRunASAP , run as soon as possible
  // std::set<TopoStruct*, comp> run_alap;  // 3, TaskClassifier::kRunALAP , run as late as possible
  const int32_t num_classifier = 4;
  std::vector<std::set<TopoStruct*, comp>> waiting_lists(num_classifier);

  std::vector<int32_t> remain_task_nums(num_classifier, 0);

  auto SetOrderInGraph = [&](TaskNode* task_node) {
    task_node->set_order_in_graph(order_in_graph);
    ordered_task_nodes->emplace_back(task_node);
    ++order_in_graph;
  };

  // wait in the list
  auto wait = [&](TaskNode* node) {
    TopoStruct* first_topo_struct = &task_node2topo_struct[node];
    // Check if all the same nodes are ready simultaneously
    TopoStruct* curr_topo_struct = first_topo_struct->next_same_node;
    while (curr_topo_struct && curr_topo_struct != first_topo_struct) {
      if (curr_topo_struct->counter) { return; }
      curr_topo_struct = curr_topo_struct->next_same_node;
    }
    // Add all the same nodes at the same time
    curr_topo_struct = first_topo_struct;
    auto& waiting_list = waiting_lists[GetTaskClassifier(node)];
    while (true) {
      waiting_list.insert(curr_topo_struct);
      // Reduce counter then this node will never be added again
      // Though inserting into a map twice does not matter because of the same keys
      curr_topo_struct->counter--;
      curr_topo_struct = curr_topo_struct->next_same_node;
      if ((!curr_topo_struct) || (curr_topo_struct == first_topo_struct)) { break; }
    }
  };

  // initialization
  task_graph->ForEachNode([&](TaskNode* node) {
    int32_t count = node->in_edges().size();
    task_node2topo_struct[node].counter = count;
    if (count == 0) { wait(node); }
    remain_task_nums[GetTaskClassifier(node)]++;
  });

  // Finish execution
  auto finish_execution = [&](TaskNode* node) {
    node->ForEachNodeOnOutEdge([&](TaskNode* out) {
      --(task_node2topo_struct[out].counter);
      if (task_node2topo_struct[out].counter == 0) { wait(out); }
    });
  };

  // Move the first node of the waiting list to the execution list
  auto move2execution_list = [&](std::set<TopoStruct*, comp>& waiting_list,
                                 std::vector<TaskNode*>& execution_list) {
    TaskNode* first_node = (*waiting_list.begin())->node;
    int32_t execution_num = 0;
    TopoStruct* first_topo_struct = &task_node2topo_struct[first_node];
    // Find all the same nodes in different machine
    // They should be run simultaneously
    TopoStruct* curr_topo_struct = first_topo_struct;
    while (true) {
      execution_num++;
      execution_list.push_back(curr_topo_struct->node);
      waiting_list.erase(curr_topo_struct);
      // move and maybe leave
      curr_topo_struct = curr_topo_struct->next_same_node;
      if ((!curr_topo_struct) || (curr_topo_struct == first_topo_struct)) { break; }
    }
    CHECK_GT(execution_num, 0) << "Error, no task nodes are moved to the execution list";
  };

  // Execute the first n nodes in the waiting list
  auto execute = [&](int32_t list_classifier, int32_t n, bool if_reverse = false) {
    // n > 0
    if (n <= 0) { return; }
    auto& waiting_list = waiting_lists[list_classifier];
    std::vector<TaskNode*> execution_list;
    int32_t count = 0;
    // Move to the execution list
    while (!waiting_list.empty()) {
      move2execution_list(waiting_list, execution_list);
      count++;
      if (count >= n) { break; }
    }
    remain_task_nums[list_classifier] -= execution_list.size();
    // Set the order and then remove from the execution list
    for (auto* node : execution_list) {
      SetOrderInGraph(node);
      finish_execution(node);
    }
  };

  // straightening
  while (true) {
    if (waiting_lists[TaskClassifier::kRunASAP].empty()) {
      if (waiting_lists[TaskClassifier::kWaitingTransfer].empty()) {
        if (waiting_lists[TaskClassifier::kWaitingComputation].empty()) {
          if (waiting_lists[TaskClassifier::kRunALAP].empty()) {
            // All the waiting lists are empty
            break;
          } else {
            // Execute all the nodes left
            execute(TaskClassifier::kRunALAP, waiting_lists[TaskClassifier::kRunALAP].size());
          }
        } else {
          // Execute one computation node
          execute(TaskClassifier::kWaitingComputation, 1);
        }
      } else {
        int32_t computation_num =
            std::min(int32_t(waiting_lists[TaskClassifier::kWaitingComputation].size()
                             / (waiting_lists[TaskClassifier::kWaitingTransfer].size())),
                     remain_task_nums[TaskClassifier::kWaitingComputation]
                         / remain_task_nums[TaskClassifier::kWaitingTransfer]);
        // Holding the transfer
        std::vector<TaskNode*> transfer_execution_list;
        move2execution_list(waiting_lists[TaskClassifier::kWaitingTransfer],
                            transfer_execution_list);
        remain_task_nums[TaskClassifier::kWaitingTransfer] -= transfer_execution_list.size();
        for (auto* transfer_node : transfer_execution_list) { SetOrderInGraph(transfer_node); }
        // Overlap transfer with computation
        execute(TaskClassifier::kWaitingComputation, computation_num);

        // Release the transfer
        for (auto* transfer_node : transfer_execution_list) { finish_execution(transfer_node); }
      }
    } else {
      execute(TaskClassifier::kRunASAP, waiting_lists[TaskClassifier::kRunASAP].size());
    }
  }
}

}  // namespace oneflow
