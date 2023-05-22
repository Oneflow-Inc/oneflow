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

// Straighten a subset of the op graph
void StraightenMemorySubGraph(const std::vector<const OpNode*>& sub_graph,
                              std::vector<const OpNode*>* ordered_op_nodes);

// Straighten the whole op graph
void StraightenMemoryOpGraph(const OpGraph& op_graph, std::vector<const OpNode*>* ordered_op_nodes);

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_STRAIGHTEN_MEMORY_H_
