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

#include "oneflow/core/auto_parallel/auto_memory.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/graph/straighten_nodes.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {
namespace auto_parallel {

namespace {

class TopoStruct {
 public:
  SbpNode* sbp_node = nullptr;
  OpNode* op_node = nullptr;
  int64_t memory_increment = -1;
  int32_t activation_time = -1;
  bool is_reusable = false;

  // Decide whether all the produced registers are reusable
  void ComputeIsReusable();
  // Memory increment = (memory of out registers) - (memory of in registers)
  void InitMemoryIncrement();
  // Activation time = time of cpu - time of gpu
  void ComputeActivationTime();

  // deciding parameter
  // kTributaryLayerAscend = 0,     // small tributary layers go first
  // kDistanceToOverlapAscend = 1,  // small minimum distance to overlap go first
  // kLayerAscend = 2,              // first in first out
  // kMemoryIncrementAscend = 3,    // small memory increment go first
  // kActivationTimeAscend = 4,     // small activation time go first
  // kTributaryLayerDescend = 100,     // large tributary layers go first
  // kDistanceToOverlapDescend = 101,  // long distance to overlap go first
  // kLayerDescend = 102,              // last in first out
  // kMemoryIncrementDescend = 103,    // large memory increment go first
  // kActivationTimeDescend = 104,     // large activation time go first
  int64_t GetDecidingParameter(StraightenOrder so) const;
};

std::vector<StraightenOrder> decide_parameters;

bool IsProducedRegisterReusable(const Operator& op) {
  // The repeat, acc, pack and unpack operators have non-reusable registers
  // and a -1 register num at this moment.
  if (op.op_conf().has_user_conf()) {
    const auto& op_type_name = op.op_conf().user_conf().op_type_name();
    // We record the frequency in swin-transformer on the right hand side
    // and adjust the position accordingly.
    if (op_type_name == "repeat"     // 213
        || op_type_name == "acc"     // 173
        || op_type_name == "unpack"  // 2
        || op_type_name == "pack"    // 1
    ) {
      return false;
    }
  }
  // NOTE: Please refer to oneflow/core/graph_impl/normal_forward_compute_task_node.cpp
  // NormalForwardCompTaskNode::ProduceOutRegstByNameAndBlockNum
  // for detail.
  // We can not use <= 0 here since RegstNum4Op returns a number with type size_t.
  // -1 is actually 18446744073709551615 here.
  return RegstNum4Op(op) == -1;
}

// deciding parameter
// kTributaryLayerAscend = 0,     // small tributary layers go first
// kDistanceToOverlapAscend = 1,  // small minimum distance to overlap go first
// kLayerAscend = 2,              // first in first out
// kMemoryIncrementAscend = 3,    // small memory increment go first
// kActivationTimeAscend = 4,     // small activation time go first
// kTributaryLayerDescend = 100,     // large tributary layers go first
// kDistanceToOverlapDescend = 101,  // long distance to overlap go first
// kLayerDescend = 102,              // last in first out
// kMemoryIncrementDescend = 103,    // large memory increment go first
// kActivationTimeDescend = 104,     // large activation time go first
int64_t TopoStruct::GetDecidingParameter(StraightenOrder so) const {
  int64_t sign = 1;
  if (so >= kDiff4AscendDescend) {
    so = StraightenOrder(int(so) - kDiff4AscendDescend);
    sign = -1;
  }
  switch (so) {
    case StraightenOrder::kTributaryLayerAscend: return sign * sbp_node->GetTributaryLayer();
    case StraightenOrder::kDistanceToOverlapAscend: return 0;
    case StraightenOrder::kLayerAscend: return sign * sbp_node->GetMinLayer();
    case StraightenOrder::kMemoryIncrementAscend: return sign * memory_increment;
    case StraightenOrder::kActivationTimeAscend: return sign * activation_time;
    default: return 0;
  }
}

// Memory increment = (memory of out registers) - (memory of in registers)
// It only contains the first term
void TopoStruct::InitMemoryIncrement() {
  if (memory_increment < 0) {
    memory_increment = 0;
    const auto& curr_operator = op_node->op();
    if (is_reusable) {
      for (const auto& obn : curr_operator.output_bns()) {
        const LogicalBlobId& lbi = curr_operator.BnInOp2Lbi(obn);
        const BlobDesc& logical_blob_desc = op_node->LogicalBlobDesc4Lbi(lbi);
        memory_increment += TotalByteSize4BlobDesc(logical_blob_desc);
      }
    }
  }
}

// Activation time = time of cpu - time of gpu
void TopoStruct::ComputeActivationTime() {
  if (op_node != nullptr && LongerActivationTimeInCpu(op_node->op().op_conf())) {
    activation_time = 1;
  } else {
    activation_time = 0;
  }
}

void TopoStruct::ComputeIsReusable() { is_reusable = IsProducedRegisterReusable(op_node->op()); }

void ComputeAllMemoryIncrement(std::vector<TopoStruct*>& topo_structs) {
  // Construct the map from a lbi to its consumers
  HashMap<LogicalBlobId, std::vector<TopoStruct*>> lbi2consumer_topo_structs;
  for (auto& topo_struct : topo_structs) {
    const auto& consumer = topo_struct->op_node->op();
    for (const auto& ibn : consumer.input_bns()) {
      const LogicalBlobId& lbi = consumer.BnInOp2Lbi(ibn);
      lbi2consumer_topo_structs[lbi].push_back(topo_struct);
    }
  }
  // Compute the memory increment for produced blobs
  for (auto& topo_struct : topo_structs) { topo_struct->InitMemoryIncrement(); }
  // Subtract the consumed memory
  for (auto& pair : lbi2consumer_topo_structs) {
    int64_t memory_decrease =
        TotalByteSize4BlobDesc(pair.second[0]->op_node->LogicalBlobDesc4Lbi(pair.first))
        / pair.second.size();
    for (auto& consumer_topo_struct : pair.second) {
      consumer_topo_struct->memory_increment -= memory_decrease;
    }
  }
}

}  // anonymous namespace

}  // namespace auto_parallel
}  // namespace oneflow
