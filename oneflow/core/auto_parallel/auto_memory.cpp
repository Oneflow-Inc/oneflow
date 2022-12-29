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
  // Memory increment = (memory of out registers) - (memory of in registers)
  int64_t memory_increment = -1;
  int32_t exceed_time = -1;
  bool is_reusable = false;
  int32_t counter = 0;

  explicit TopoStruct(SbpNode* sbp_node_);

  // Decide whether all the produced registers are reusable
  void ComputeIsReusable();
  // Exceed time = time of cpu - time of gpu
  void ComputeExceedTime();

  // deciding parameter
  // kTributaryLayerAscend = 0,     // small tributary layers go first
  // kDistanceToOverlapAscend = 1,  // small minimum distance to overlap go first
  // kLayerAscend = 2,              // first in first out
  // kMemoryIncrementAscend = 3,    // small memory increment go first
  // kExceedTimeAscend = 4,         // small exceed time go first
  // kTributaryLayerDescend = 100,     // large tributary layers go first
  // kDistanceToOverlapDescend = 101,  // long distance to overlap go first
  // kLayerDescend = 102,              // last in first out
  // kMemoryIncrementDescend = 103,    // large memory increment go first
  // kExceedTimeDescend = 104,         // large exceed time go first
  int64_t GetDecidingParameter(StraightenOrder so) const;
};

static StraightenAlgorithmTag sat;

static std::vector<StraightenOrder> decide_parameters;

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

TopoStruct::TopoStruct(SbpNode* sbp_node_)
    : sbp_node(sbp_node_), op_node(sbp_node_->GetOperatorNode()) {
  ComputeIsReusable();
  ComputeExceedTime();
}

// deciding parameter
// kTributaryLayerAscend = 0,     // small tributary layers go first
// kDistanceToOverlapAscend = 1,  // small minimum distance to overlap go first
// kLayerAscend = 2,              // first in first out
// kMemoryIncrementAscend = 3,    // small memory increment go first
// kExceedTimeAscend = 4,         // small exceed time go first
// kTributaryLayerDescend = 100,     // large tributary layers go first
// kDistanceToOverlapDescend = 101,  // long distance to overlap go first
// kLayerDescend = 102,              // last in first out
// kMemoryIncrementDescend = 103,    // large memory increment go first
// kExceedTimeDescend = 104,         // large exceed time go first
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
    case StraightenOrder::kExceedTimeAscend: return sign * exceed_time;
    default: return 0;
  }
}

// Exceed time = time of cpu - time of gpu
void TopoStruct::ComputeExceedTime() {
  if (ShortGpuTime(op_node->op().op_conf())) {
    exceed_time = 1;
  } else {
    exceed_time = 0;
  }
}

void TopoStruct::ComputeIsReusable() { is_reusable = IsProducedRegisterReusable(op_node->op()); }

// Compute the memory increment for all the topological structures
void ComputeAllMemoryIncrement(std::vector<TopoStruct*>& topo_structs,
                               HashMap<LogicalBlobId, int32_t>& lbi2id,
                               std::vector<std::vector<TopoStruct*>>& id2consumer_topo_structs,
                               std::vector<int64_t>& id2blob_size) {
  // Compute the memory increment for produced blobs
  for (auto& topo_struct : topo_structs) {
    topo_struct->memory_increment = 0;
    const auto& curr_operator = topo_struct->op_node->op();
    if (topo_struct->is_reusable) {
      for (const auto& obn : curr_operator.output_bns()) {
        const LogicalBlobId& lbi = curr_operator.BnInOp2Lbi(obn);
        auto it = lbi2id.find(lbi);
        if (it == lbi2id.end()) {
          // There exist some blobs that do not have any consumer
          // Such as: op name:
          // model.cls_head.loss_func.lm_loss-sparse_softmax_cross_entropy_ms-231-split_softmax_reduce_max_global_stage
          // blob name: mask_0
          const BlobDesc& logical_blob_desc = topo_struct->op_node->LogicalBlobDesc4Lbi(lbi);
          lbi2id[lbi] = id2blob_size.size();
          id2blob_size.push_back(TotalByteSize4BlobDesc(logical_blob_desc));
          // There are some inconsistency between id2blob_size and id2consumer_topo_structs
          // We would deal with that at the end to avoid division by 0
          topo_struct->memory_increment += id2blob_size.back();
        } else {
          topo_struct->memory_increment += id2blob_size[it->second];
        }
      }
    }
  }
  // Subtract the consumed memory
  for (int32_t index = 0; index < id2consumer_topo_structs.size(); index++) {
    int64_t memory_decrease = id2blob_size[index] / id2consumer_topo_structs[index].size();
    for (auto& consumer_topo_struct : id2consumer_topo_structs[index]) {
      consumer_topo_struct->memory_increment -= memory_decrease;
    }
  }
  // Add empty vectors for all those blobs without consumers
  id2consumer_topo_structs.resize(id2blob_size.size());
}

}  // anonymous namespace

void InitMemory(SbpGraph* sbp_graph, bool nccl_use_compute_stream) {
  // Generate topological data structure for each sbp node
  HashMap<SbpNode*, TopoStruct> sbp_node2topo_struct;
  std::vector<TopoStruct*> topo_structs;
  // Traverse all the nodes in the sbp graph
  for (const auto& sbp_node : sbp_graph->GetNodeList()) {
    CHECK(sbp_node->GetOperatorNode() != nullptr)
        << "No proxy node allow at this status. InitMemory() should be run before sbp collector!";
    sbp_node2topo_struct.insert({sbp_node, TopoStruct(sbp_node)});
    topo_structs.push_back(&sbp_node2topo_struct.at(sbp_node));
  }

  // Construct the map from a lbi to its id, consumers, blob size
  HashMap<LogicalBlobId, int32_t> lbi2id;
  std::vector<std::vector<TopoStruct*>> id2consumer_topo_structs;
  std::vector<int64_t> id2blob_size;
  for (auto& topo_struct : topo_structs) {
    const auto& consumer = topo_struct->op_node->op();
    for (const auto& ibn : consumer.input_bns()) {
      const LogicalBlobId& lbi = consumer.BnInOp2Lbi(ibn);
      auto it = lbi2id.find(lbi);
      if (it == lbi2id.end()) {
        lbi2id[lbi] = id2blob_size.size();
        const BlobDesc& logical_blob_desc = topo_struct->op_node->LogicalBlobDesc4Lbi(lbi);
        id2blob_size.push_back(TotalByteSize4BlobDesc(logical_blob_desc));
        id2consumer_topo_structs.push_back({topo_struct});
      } else {
        id2consumer_topo_structs[it->second].push_back(topo_struct);
      }
    }
  }
  // Compute the memory increment for all the topological structures
  ComputeAllMemoryIncrement(topo_structs, lbi2id, id2consumer_topo_structs, id2blob_size);

  // Update sat, since sat might be changed in previous jobs
  UpdateSat(sbp_node2topo_struct, &sat);

  // Decide which node should run first
  InitDecideParameters(sat, &decide_parameters);
  VLOG(3) << "Straightening order in sbp graph: ";
  for (int32_t decide_parameter : decide_parameters) { VLOG(3) << decide_parameter; }

  // Order in the waiting sets
  struct comp {
    bool operator()(const TopoStruct* a, const TopoStruct* b) const {
      for (auto decide_parameter : decide_parameters) {
        auto decide_parameter_a = a->GetDecidingParameter(decide_parameter);
        auto decide_parameter_b = b->GetDecidingParameter(decide_parameter);
        if (decide_parameter_a != decide_parameter_b) {
          return decide_parameter_a < decide_parameter_b;
        }
      }
      return a->op_node->op().op_name() < b->op_node->op().op_name();
    }
  };
  std::set<TopoStruct*, comp> waiting_list;

  // Order of execution for topological structs
  std::vector<TopoStruct*> ordered_topo_structs;

  // Wait in the list
  auto wait = [&](SbpNode* sbp_node) { waiting_list.insert(&sbp_node2topo_struct.at(sbp_node)); };

  // Initialization
  for (auto& topo_struct : topo_structs) {
    topo_struct->counter = topo_struct->sbp_node->GetEdgesIn().size();
    if (topo_struct->counter == 0) { wait(topo_struct->sbp_node); }
  }

  // Finish execution
  auto finish_execution = [&](SbpNode* sbp_node) {
    for (const auto& edge_out : sbp_node->GetEdgesOut()) {
      SbpNode* end_node = edge_out->GetEndNode();
      int32_t& end_node_counter = sbp_node2topo_struct.at(end_node).counter;
      --end_node_counter;
      if (end_node_counter == 0) { wait(end_node); }
    }
  };

  // Execute the first node in the waiting list
  // Make sure to check that waiting list is not empty before execution
  auto execute = [&]() {
    auto first_topo_struct = *waiting_list.begin();
    // Set the order of execution for sbp nodes
    ordered_topo_structs.push_back(first_topo_struct);
    waiting_list.erase(waiting_list.begin());
    finish_execution(first_topo_struct->sbp_node);
  };

  // straightening
  while (!waiting_list.empty()) { execute(); }

  // Mark the memory support, which contains two part:
  // All the non-reusable memory and those blobs which is a part of the maximum reusable memory
  int64_t max_reusable_memory = 0;
  int64_t curr_reusable_memory = 0;
  std::vector<int32_t> id2count(id2blob_size.size(), -1);
  // Blobs born, increase count and memory
  auto GenerateBlobs = [&](TopoStruct* topo_struct) {
    const auto& curr_operator = topo_struct->op_node->op();
    if (topo_struct->is_reusable) {
      for (const auto& obn : curr_operator.output_bns()) {
        const LogicalBlobId& lbi = curr_operator.BnInOp2Lbi(obn);
        int32_t index = lbi2id.at(lbi);
        // Reusable blobs born
        curr_reusable_memory += id2blob_size[index];
        id2count[index] = id2consumer_topo_structs[index].size();
      }
    }
  };
  // Blobs die, decrease count and memory
  auto KillBlobs = [&](TopoStruct* topo_struct) {
    const auto& curr_operator = topo_struct->op_node->op();
    // Those reusable blobs who do not have a consumer would die immediately
    // For example:
    // register_num: 1, op_name:
    // "model.cls_head.loss_func.lm_loss-sparse_softmax_cross_entropy_ms-231-split_softmax_reduce_max_device_stage",
    // blob_name: "mask_0", shape { dim: 2048 dim: 21248 },
    // data_type: kBool, time_shape { dim: 1 dim: 1 }, enable_reuse_mem: true,
    // alloc_before_actor: 369, free_after_actor: 369
    if (topo_struct->is_reusable) {
      for (const auto& obn : curr_operator.output_bns()) {
        const LogicalBlobId& lbi = curr_operator.BnInOp2Lbi(obn);
        int32_t index = lbi2id.at(lbi);
        // Do not have consumer
        if (id2count[index] == 0) {
          // Reusable blobs die
          curr_reusable_memory -= id2blob_size[index];
        }
      }
    }
    // Reduce the counter and kill the blobs if count to 0
    for (const auto& ibn : curr_operator.input_bns()) {
      const LogicalBlobId& lbi = curr_operator.BnInOp2Lbi(ibn);
      int32_t index = lbi2id.at(lbi);
      if (id2count[index] > 0) {
        --id2count[index];
        if (id2count[index] == 0) {
          // Reusable blobs die
          curr_reusable_memory -= id2blob_size[index];
        }
      }
    }
  };
  // Calculate the maximum reusable memory and mark those fixed memory
  for (auto& topo_struct : ordered_topo_structs) {
    // Blobs born, increase count and memory
    GenerateBlobs(topo_struct);
    // Record the maximum memory
    if (curr_reusable_memory > max_reusable_memory) { max_reusable_memory = curr_reusable_memory; }
    // Blobs die, decrease count and memory
    KillBlobs(topo_struct);
  }

  // Make sure that every blob dies
  CHECK_EQ(curr_reusable_memory, 0) << " Have not kill all the reusable blobs!";

  // Mark those reusable memory which constitute the maximum reusable memory
  for (auto& topo_struct : ordered_topo_structs) {
    // Blobs born, increase count and memory
    GenerateBlobs(topo_struct);
    // Mark the first found support
    if (curr_reusable_memory == max_reusable_memory) {
      // Mark the temporary memory created by this operator
      if (topo_struct->is_reusable) {
        const auto& curr_operator = topo_struct->op_node->op();
        for (const auto& obn : curr_operator.output_bns()) {
          const LogicalBlobId& lbi = curr_operator.BnInOp2Lbi(obn);
          int32_t index = lbi2id.at(lbi);
          // We would use id2count != 0 to record the lbi support
          // Those obn with no consumers have id2count[index] == 0, now it would be set to 1
          id2count[index] = 1;
        }
      }
      // The other lbi in the support would have a non-zero id2count
      // No further process needed
      break;
    }
    // Blobs die, decrease count and memory
    KillBlobs(topo_struct);
  }

  // Initialize memory for each sbp node
  for (auto& topo_struct : topo_structs) {
    topo_struct->sbp_node->InitializeMemory(topo_struct->is_reusable, lbi2id, id2count,
                                            nccl_use_compute_stream);
  }
}

}  // namespace auto_parallel
}  // namespace oneflow
