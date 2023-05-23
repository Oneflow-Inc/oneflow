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
#ifndef ONEFLOW_CORE_AUTO_PARALLEL_STRAIGHTEN_MEMORY_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_STRAIGHTEN_MEMORY_H_

#include "oneflow/core/auto_parallel/sbp_graph.h"
#include "oneflow/core/graph/op_graph.h"
namespace oneflow {

namespace auto_parallel {

class NoCleaningMarkerAncestor {
 public:
  static int32_t marker;
  int32_t status = 0;

  bool IfMarked() const { return status == marker; }
  bool IfNotMarked() const { return status != marker; }
  void Mark() { status = marker; }
  void UnMark() { status = 0; }
};

void ResetNoCleaningMarkerAncestor();
void InitNoCleaningMarkerAncestor();

class NoCleaningMarkerDescendant {
 public:
  static int32_t marker;
  int32_t status = 0;

  bool IfMarked() const { return status == marker; }
  bool IfNotMarked() const { return status != marker; }
  void Mark() { status = marker; }
  void UnMark() { status = 0; }
};

void ResetNoCleaningMarkerDescendant();
void InitNoCleaningMarkerDescendant();

class MemoryTopoStruct {
 public:
  // Memory increment = (memory of out registers) - (memory of in registers)
  int64_t memory_increment = -1;
  int64_t peak_memory = -1;
  // max difference = peak memory - final memory increment
  int64_t max_difference = 0;
  int32_t min_layer = -1;
  int32_t blob_id = -1;
  // Blocking means you must execute this node before executing any other nodes in the set
  HashSet<MemoryTopoStruct*> blocking_topo_structs;
  int32_t blocking_count = -1;
  // executed means that it has been executed
  bool executed = false;
  // Accumulate memory increment of all the necessary topological structures
  int64_t accumulate_memory_increment = 0;
  int64_t peak_memory_during_accumulation = 0;
  int64_t max_difference_during_accumulation = 0;
  // Whether visited during memory accumulating
  NoCleaningMarkerAncestor visited_ancestors;
  // Whether visited while finding descendants
  NoCleaningMarkerDescendant visited_descendant;
  // waiting in the map before execution
  bool waiting = false;

  HashSet<MemoryTopoStruct*> in_topo_structs;
  HashSet<MemoryTopoStruct*> out_topo_structs;

  // The topo structs to be executed in a reverse order right before this topo struct
  // For example:
  // This topo struct: A, Pre topo structs: {B, C, D}
  // This topo struct: B, Pre topo structs: {E}
  // This topo struct: D, Pre topo structs: {F, G}
  // And the graph is: H -> A -> I
  // Then the execution order is H, G, F, D, C, E, B, A, I
  std::vector<MemoryTopoStruct*> pre_topo_structs;
  // The topo structs to be executed immediately after this topo struct
  std::vector<MemoryTopoStruct*> post_topo_structs;

  // Execute the positive ancestors in order with the smallest peak memory
  std::vector<MemoryTopoStruct*> ordered_ancestors;

  explicit MemoryTopoStruct(int32_t blob_id_) : blob_id(blob_id_){};

  // Compute the minimum layer of this node
  int32_t ComputeMinLayer();
  // Block the descendants with negative memory increment
  void BlockDescendants();

  void SetAccumulateMemoryIncrement();
  void MarkAncestors();
  int64_t SingleNodePriority();
  int64_t AccumulationPriority();

  void MarkDescendantFromThis2Layer(int32_t max_layer);

 private:
  void VisitAncestorsAndItself(const std::function<void(MemoryTopoStruct*)>& Handle);
  // Mark all its descendant with min_layer <= max_layer
  void MarkDescendantUp2Layer(int32_t max_layer);
  // Block descendants and store the blocking nodes in the given hash set
  void BlockDescendants(HashSet<MemoryTopoStruct*>* blocking_nodes);
};

void ConnectTwoNodes(MemoryTopoStruct* producer, MemoryTopoStruct* consumer);

// The memory straighten algorithm
void StraightenMemory(std::vector<MemoryTopoStruct*>* topo_structs,
                      const std::vector<MemoryTopoStruct*>& id2producer_topo_struct,
                      std::vector<std::vector<MemoryTopoStruct*>>* id2consumer_topo_structs,
                      const std::vector<int64_t>& id2blob_size,
                      const std::vector<bool>& id2is_reusable,
                      std::vector<MemoryTopoStruct*>* ordered_topo_structs);

const int32_t kOriginNode = -123;

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_STRAIGHTEN_MEMORY_H_
