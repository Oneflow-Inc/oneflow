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
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace {

bool IsTransferNode(int32_t task_type) {
  return task_type == 12 || task_type == 13 || (48 <= task_type && task_type <= 64);
}

}  // anonymous namespace

// Drop down the maximum layer with the minimum layer form consumer
void TopoStruct::DropTributaryLayer(int32_t upper_bound) {
  if (upper_bound < TributaryLayer || TributaryLayer < 0) { TributaryLayer = upper_bound; }
}

// Should initialize the counter to be the number of out edges
// Compute maximum layer for tributaries
void TopoStruct::SpreadTributaryLayer(HashMap<TaskNode*, TopoStruct>& task_node2topo_struct) {
  if (counter || MinLayer <= 0) { return; }
  int32_t producer_max_lay = 0;
  if (IfMainstem) {
    producer_max_lay = MinLayer - 1;
  } else {
    // On a tributary, the operator could be run later.
    producer_max_lay = TributaryLayer;
    // producer_max_lay = TributaryLayer - 1;
  }
  node->ForEachNodeOnInEdge([&](TaskNode* in) {
    auto& topo_struct_in = task_node2topo_struct[in];
    topo_struct_in.DropTributaryLayer(producer_max_lay);
    if (--topo_struct_in.counter == 0) {
      topo_struct_in.SpreadTributaryLayer(task_node2topo_struct);
    }
  });
  // Reduce counter to -1 to avoid visitting again
  counter--;
}

// Judge if this node is on the mainstem
// If so, judge it for its producer/upstream nodes
void TopoStruct::SpreadMainstem(HashMap<TaskNode*, TopoStruct>& task_node2topo_struct) {
  // Skip it if this node is already judged.
  if (IfMainstem) { return; }
  CHECK(MinLayer >= 0) << "TopoStruct not initialized!";
  IfMainstem = true;
  // If I am in the mainstem, then all the children with (MinLayer >= my layer id - 1) would be
  // considered as in the mainstem
  node->ForEachNodeOnInEdge([&](TaskNode* in) {
    auto& topo_struct_in = task_node2topo_struct[in];
    if (topo_struct_in.MinLayer == MinLayer - 1) {
      topo_struct_in.SpreadTributaryLayer(task_node2topo_struct);
    }
  });
}

// The minimum computation distance from the beginning of this op to the next transfer
int32_t TopoStruct::GetMinDistance2Transfer(HashMap<TaskNode*, TopoStruct>& task_node2topo_struct) {
  if (MinDistance2Transfer >= 0) { return MinDistance2Transfer; }
  // if this node is a transfer node
  if (IsTransferNode(node->GetTaskType())) {
    MinDistance2Transfer = 0;
    return MinDistance2Transfer;
  }
  MinDistance2Transfer = 1000000;
  node->ForEachNodeOnOutEdge([&](TaskNode* out) {
    MinDistance2Transfer =
        std::min(MinDistance2Transfer,
                 task_node2topo_struct[out].GetMinDistance2Transfer(task_node2topo_struct));
  });
  return ++MinDistance2Transfer;
}

// deciding parameter
int32_t TopoStruct::GetDecidingParameter(int32_t i) const {
  int32_t sign = 1;
  if (i >= 3) {
    i -= 3;
    sign = -1;
  }
  switch (i) {
    case 0: return sign * TributaryLayer;
    case 1: return sign * MinDistance2Transfer;
    case 2: return sign * MinLayer;
  }
  return 0;
}

// Find the mianstem of the task graph, then reduce the wait time for tributaries
void FindMainstem(HashMap<TaskNode*, TopoStruct>& task_node2topo_struct) {
  // Find the maximum layer number
  int32_t max_MinLayer = -1;
  for (const auto& pair : task_node2topo_struct) {
    if (max_MinLayer < pair.second.MinLayer) { max_MinLayer = pair.second.MinLayer; }
  }
  // All the nodes with MinLayer>=mainstem_end_id would be considered as mainstem nodes
  int32_t mainstem_end_id = max_MinLayer - 4;
  for (auto& pair : task_node2topo_struct) {
    auto& topo_struct = pair.second;
    // Initialize the counter and Tributary Layer
    topo_struct.counter = pair.first->out_edges().size();
    topo_struct.TributaryLayer = max_MinLayer;
    // Find out all the nodes on the mainstem.
    if (topo_struct.MinLayer >= mainstem_end_id) {
      topo_struct.SpreadMainstem(task_node2topo_struct);
    }
  }

  for (auto& pair : task_node2topo_struct) {
    // Compute maximum layer for tributaries
    pair.second.SpreadTributaryLayer(task_node2topo_struct);
    // Set the MinDistance2Transfer for each topological structure
    pair.second.GetMinDistance2Transfer(task_node2topo_struct);
  }
}

void StraightenNodes(TaskGraph* task_graph, std::vector<TaskNode*>& ordered_task_nodes) {
  // The function for settle the order in the graph
  int64_t order_in_graph = 0;
  HashMap<int32_t, int32_t> task_type_map;

  // The same order in the set
  // It is only run in the following situation because there are too many implicit conditions.
  auto should_run_simultaneously = [](TopoStruct* a, TopoStruct* b) -> bool {
    // Normal node would have the same name
    if (a->node->GetTaskType() == 1) { return a->node->VisualStr() == b->node->VisualStr(); }
    // Otherwise they must have the same parameters with different machine ids and the closest node
    // id. We only use Min Layer here, since Tributary Layer might be different due to asymmetry of
    // graph.
    return true;
  };

  // move the head from source to target
  auto move_front_between_maps = [](std::map<int32_t, TopoStruct*>& source,
                                    std::map<int32_t, TopoStruct*>& target) {
    if (!source.empty()) {
      const auto& front = source.begin();
      target[front->first] = front->second;
      source.erase(front);
    }
  };

  // Generate topological data structure for each task node
  HashMap<TaskNode*, TopoStruct> task_node2topo_struct;
  // Determine the same nodes which should run simultaneously
  HashMap<int32_t, HashMap<int32_t, std::map<int32_t, TopoStruct*>>>
      task_type2machine_id2node_id2topo_structs;
  std::map<int32_t, TopoStruct*> min_node_id2topo_struct;
  int32_t previous_MinLayer = 0;
  task_graph->TopoForEachNodeFast([&](TaskNode* node) {
    auto& topo_struct = task_node2topo_struct[node];
    topo_struct.node = node;
    if (node->in_edges().empty()) {
      topo_struct.MinLayer = 0;
    } else {
      int32_t max_min_layer = 0;
      node->ForEachNodeOnInEdge([&](TaskNode* in) {
        max_min_layer = std::max(max_min_layer, task_node2topo_struct[in].MinLayer);
      });
      topo_struct.MinLayer = max_min_layer + 1;
      // Deal with all the nodes with MinLayer=previous_MinLayer
      std::cout << "Min Layer: " << topo_struct.MinLayer << std::endl;
      if (max_min_layer >= previous_MinLayer) {
        // Using "7" to represent "and"
        // a7b means a pair (a, b)
        for (auto& task_type7machine_id2node_id2topo_structs :
             task_type2machine_id2node_id2topo_structs) {
          auto& machine_id2node_id2topo_structs = task_type7machine_id2node_id2topo_structs.second;
          // Initializing the smallest node id for each machine
          for (auto& machine_id7node_id2topo_structs : machine_id2node_id2topo_structs) {
            move_front_between_maps(machine_id7node_id2topo_structs.second,
                                    min_node_id2topo_struct);
          }
          //
          while (!min_node_id2topo_struct.empty()) {
            auto* topo_struct_min_node_id = min_node_id2topo_struct.begin()->second;
            // Store the same nodes in different machine
            std::vector<TopoStruct*> same_nodes;
            for (auto& min_node_id7topo_struct : min_node_id2topo_struct) {
              auto* curr_topo_struct = min_node_id7topo_struct.second;
              // Find out all the same nodes
              if (should_run_simultaneously(topo_struct_min_node_id, curr_topo_struct)) {
                same_nodes.push_back(curr_topo_struct);
              }
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
              move_front_between_maps(
                  machine_id2node_id2topo_structs[same_node_topo_struct->node->machine_id()],
                  min_node_id2topo_struct);
            }
          }
        }
        // Renew the previous MinLayer at the end
        previous_MinLayer = topo_struct.MinLayer;
      }
    }
    // Put the topo structure into the map, waiting for determine the same nodes
    task_type2machine_id2node_id2topo_structs[node->GetTaskType()][node->machine_id()]
                                             [node->node_id()] = &topo_struct;
  });

  // Generate other parameters in the topological data structure
  FindMainstem(task_node2topo_struct);

  // test debug
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "Straightening order type: " << ParseIntegerFromEnv("Parameter0", 0) << ", "
              << ParseIntegerFromEnv("Parameter1", 1) << ", "
              << ParseIntegerFromEnv("Parameter2", 2) << std::endl;
  }
  // Order in the waiting sets
  // Decide which node should run first
  struct comp {
    bool operator()(const TopoStruct* a, const TopoStruct* b) const {
      static std::vector<int64_t> decide_parameters({ParseIntegerFromEnv("Parameter0", 0),
                                                     ParseIntegerFromEnv("Parameter1", 1),
                                                     ParseIntegerFromEnv("Parameter2", 2)});
      for (int32_t decide_parameter : decide_parameters) {
        int32_t decide_parameter_a = a->GetDecidingParameter(decide_parameter);
        int32_t decide_parameter_b = b->GetDecidingParameter(decide_parameter);
        if (decide_parameter_a != decide_parameter_b) {
          return decide_parameter_a < decide_parameter_b;
        }
      }
      return a->node->node_id() < b->node->node_id();
      // auto comp_str = a->node->VisualStr().compare(b->node->VisualStr());
      // if (comp_str == 0) {
      //   // the order does not matter right now, but we need a strict order
      //   return a < b;
      // } else {
      //   return comp_str < 0;
      // };

      // if (a->TributaryLayer == b->TributaryLayer) {
      //   if (a->MinDistance2Transfer == b->MinDistance2Transfer) {
      //     if (a->MinLayer == b->MinLayer) {
      //       // Put the task with the same names together
      //       auto comp_str = a->node->VisualStr().compare(b->node->VisualStr());
      //       if (comp_str == 0) {
      //         // the order does not matter right now, but we need a strict order
      //         return a < b;
      //       } else {
      //         return comp_str < 0;
      //       }
      //     } else {
      //       // the node that shows up first has higher priority
      //       return a->MinLayer < b->MinLayer;
      //     }
      //   } else {
      //     return a->MinDistance2Transfer < b->MinDistance2Transfer;
      //   }
      // } else {
      //   return a->TributaryLayer < b->TributaryLayer;
      // }
    }
  };

  // Classify sets for the task nodes
  // std::set<TopoStruct*, comp> waiting_transfer; // 0
  // std::set<TopoStruct*, comp> waiting_computation; // 1
  // std::set<TopoStruct*, comp> run_asap;  // 2, run as soon as possible
  // std::set<TopoStruct*, comp> run_alap;  // 3, run as late as possible
  std::vector<std::set<TopoStruct*, comp>> waiting_lists(4);

  std::vector<int32_t> remain_task_nums(4, 0);

  // Classifier for the set according to the task type
  auto set_classifier = [&](TaskNode* node) {
    // Check task.pb.h for detail
    int32_t task_type = node->GetTaskType();
    if (task_type == 1) { return 1; }
    if (task_type == 12 || task_type == 13 || (48 <= task_type && task_type <= 64)) { return 0; }
    if (task_type == 47) { return 2; }
    return 3;
  };

  auto SetOrderInGraph = [&](TaskNode* task_node) {
    if (GlobalProcessCtx::Rank() == 0) {
      auto& topo_struct = task_node2topo_struct[task_node];
      std::cout << "Execution order: " << order_in_graph << ": " << task_node->VisualStr()
                << ", node id: " << task_node->node_id() << std::endl;
      std::cout << ": task type: " << task_node->GetTaskType() << ", "
                << (task_node->parallel_ctx() == 0) << ", MinLayer: " << topo_struct.MinLayer
                << ", TributaryLayer: " << topo_struct.TributaryLayer
                << ", MinDist2Transfer: " << topo_struct.MinDistance2Transfer
                << ", machine id: " << task_node->machine_id()
                << ", thread id: " << task_node->thrd_id() << std::endl;

      if (task_type_map.find(task_node->GetTaskType()) == task_type_map.end()) {
        task_type_map[task_node->GetTaskType()] = 0;
      }
      task_type_map[task_node->GetTaskType()]++;
    }
    task_node->set_order_in_graph(order_in_graph);
    ordered_task_nodes.emplace_back(task_node);
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
    auto& waiting_list = waiting_lists[set_classifier(node)];
    while (true) {
      waiting_list.insert(curr_topo_struct);
      // Reduce counter then this node will never be added again
      // Though inserting into a map twice does not matter because of the same keys
      curr_topo_struct->counter--;
      curr_topo_struct = curr_topo_struct->next_same_node;
      if ((!curr_topo_struct) || (curr_topo_struct == first_topo_struct)) { break; }
    }
  };

  std::map<int32_t, std::map<int32_t, int32_t>> task_type2node_id2machine_id;
  // initialization
  task_graph->ForEachNode([&](TaskNode* node) {
    int32_t count = node->in_edges().size();
    task_node2topo_struct[node].counter = count;
    if (count == 0) { wait(node); }
    remain_task_nums[set_classifier(node)]++;
    task_type2node_id2machine_id[node->GetTaskType()][node->node_id()] = node->machine_id();
  });

  for (auto& task_type_group : task_type2node_id2machine_id) {
    std::cout << "task type: " << task_type_group.first << std::endl;
    int32_t pre_machine_id = -1;
    for (auto& pair : task_type_group.second) {
      std::cout << "node id: " << pair.first << ", machine id: " << pair.second << ", ? "
                << (pair.second == 0 || pair.second > pre_machine_id) << std::endl;
      pre_machine_id = pair.second;
    }
  }

  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "Total task nums:" << std::endl;
    std::cout << "Transfers: " << remain_task_nums[0] << ", Computation: " << remain_task_nums[1]
              << ", Run Asap: " << remain_task_nums[2] << ", Run Alap: " << remain_task_nums[3]
              << std::endl;
  }

  // Finish execution
  auto finish_execution = [&](TaskNode* node) {
    node->ForEachNodeOnOutEdge([&](TaskNode* out) {
      if (--(task_node2topo_struct[out].counter) == 0) { wait(out); }
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
    // n>=1
    if (n <= 0) { return; }
    if (GlobalProcessCtx::Rank() == 0) {
      std::cout << "Total task nums:" << std::endl;
      std::cout << "Transfers: " << waiting_lists[0].size()
                << ", Computation: " << waiting_lists[1].size()
                << ", Run Asap: " << waiting_lists[2].size()
                << ", Run Alap: " << waiting_lists[3].size() << std::endl;
    }
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

  // int32_t max_overlap_computation_num = ParseIntegerFromEnv("MAX_OVERLAP_NUM", 40);

  // straightening
  while (true) {
    if (waiting_lists[2].empty()) {
      if (waiting_lists[0].empty()) {
        if (waiting_lists[1].empty()) {
          if (waiting_lists[3].empty()) {
            if (GlobalProcessCtx::Rank() == 0) { std::cout << "Execution done" << std::endl; }
            break;
          } else {
            execute(3, waiting_lists[3].size());
          }
        } else {
          execute(1, 1);
        }
      } else {
        int32_t computation_num =
            std::min(int32_t(waiting_lists[1].size() / (waiting_lists[0].size())),
                     remain_task_nums[1] / remain_task_nums[0]);
        // Holding the transfer
        std::vector<TaskNode*> transfer_execution_list;
        move2execution_list(waiting_lists[0], transfer_execution_list);
        remain_task_nums[0] -= transfer_execution_list.size();
        for (auto* transfer_node : transfer_execution_list) { SetOrderInGraph(transfer_node); }
        // Overlap transfer with computation
        execute(1, computation_num);

        // Release the transfer
        for (auto* transfer_node : transfer_execution_list) { finish_execution(transfer_node); }
      }
    } else {
      execute(2, waiting_lists[2].size());
    }
  }

  // test debug
  if (GlobalProcessCtx::Rank() == 0) {
    std::cout << "Print all task type: " << std::endl;
    for (auto& pair : task_type_map) {
      std::cout << "task type: " << pair.first << ", " << pair.second << std::endl;
    }
  }
}

}  // namespace oneflow
