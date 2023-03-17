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
#include "oneflow/core/auto_parallel/sbp_constructor.h"
#include "oneflow/core/common/hash_container.h"
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
  const OpNode* op_node = nullptr;
  // Memory increment = (memory of out registers) - (memory of in registers)
  int64_t memory_increment = -1;
  int32_t exceed_time = -1;
  bool is_reusable = false;
  int32_t counter = 0;
  int32_t min_layer = -1;
  // The maximum min_layer among out_topo_structs
  int32_t max_layer = -1;
  // TODO: remove tributary layer
  // This node should be finished before tributary layer
  int32_t tributary_layer = -1;

  HashSet<TopoStruct*> in_topo_structs;
  HashSet<TopoStruct*> out_topo_structs;

  explicit TopoStruct(SbpNode* sbp_node_);
  explicit TopoStruct(const OpNode* op_node_);

  // Compute the minimum layer of this node
  int32_t ComputeMinLayer();
  // Compute the maximum layer of this node
  void ComputeMaxLayer(int32_t max_min_layer);
  // Compute the tributary layer
  int32_t ComputeTributaryLayer(int32_t max_min_layer);
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

TopoStruct::TopoStruct(const OpNode* op_node_) : op_node(op_node_) {
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
    case StraightenOrder::kTributaryLayerAscend: return sign * tributary_layer;
    case StraightenOrder::kDistanceToOverlapAscend: return 0;
    case StraightenOrder::kLayerAscend: return sign * min_layer;
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

// Compute the minimum layer of this node
int32_t TopoStruct::ComputeMinLayer() {
  if (min_layer >= 0) { return min_layer; }
  for (auto& in_topo_struct : in_topo_structs) {
    min_layer = std::max(min_layer, in_topo_struct->ComputeMinLayer());
  }
  return ++min_layer;
}

// Compute the maximum layer of this node
void TopoStruct::ComputeMaxLayer(int32_t max_min_layer) {
  // Execute those optimizer as soon as possible to release the register of weight_diff
  if (out_topo_structs.empty()) {
    max_layer = min_layer;
    return;
  }
  max_layer = max_min_layer;
  for (auto& out_topo_struct : out_topo_structs) {
    if (max_layer > out_topo_struct->min_layer) { max_layer = out_topo_struct->min_layer; }
  }
  --max_layer;
}

// Compute the tributary layer
int32_t TopoStruct::ComputeTributaryLayer(int32_t max_min_layer) {
  if (tributary_layer >= 0) { return tributary_layer; }
  tributary_layer = max_min_layer;
  for (auto& out_topo_struct : out_topo_structs) {
    if (tributary_layer > out_topo_struct->ComputeTributaryLayer(max_min_layer)) {
      tributary_layer = out_topo_struct->tributary_layer;
    }
  }
  return --tributary_layer;
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

void UpdateSat(const std::vector<TopoStruct*>& topo_structs, StraightenAlgorithmTag* sat) {
  *sat = GlobalJobDesc().job_conf().straighten_algorithm_tag_in_task_graph();
  if (*sat == StraightenAlgorithmTag::kOverlap4CpuGpu) {
    // If not cpu nodes, then the overlap strategy between cpu and gpu might consume large memory
    bool exist_cpu_nodes = false;
    for (const auto& topo_struct : topo_structs) {
      // Found a cpu node
      if (topo_struct->exceed_time == 1) {
        exist_cpu_nodes = true;
        break;
      }
    }
    if (!exist_cpu_nodes) {
      // Switch to the compress memory strategy, the default one
      // Since the overlap strategy for transfer might not be working on 1n1d.
      *sat = StraightenAlgorithmTag::kCompressMemory;
    }
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
      if (it != op_name2topo_structs.end()) {
        this_topo_struct->in_topo_structs.insert(it->second);
        it->second->out_topo_structs.insert(this_topo_struct);
      }
    });
    // Initialize input nodes for control edges
    for (const auto& ctrl_in_op_name : node->op().op_conf().ctrl_in_op_name()) {
      auto it = op_name2topo_structs.find(ctrl_in_op_name);
      if (it != op_name2topo_structs.end()) {
        auto& ctrl_in_topo_struct = it->second;
        this_topo_struct->in_topo_structs.insert(ctrl_in_topo_struct);
        // Initialize output nodes for this control edge simultaneously
        ctrl_in_topo_struct->out_topo_structs.insert(this_topo_struct);
      }
    }
  }
}

void ComputeLayer(std::vector<TopoStruct*>* topo_structs) {
  int32_t max_min_layer = -1;
  // Compute the minimum layer for the whole graph
  for (auto& topo_struct : *topo_structs) {
    if (max_min_layer < topo_struct->ComputeMinLayer()) { max_min_layer = topo_struct->min_layer; }
  }
  max_min_layer++;
  // Compute the maximum layer for the whole graph
  for (auto& topo_struct : *topo_structs) { topo_struct->ComputeMaxLayer(max_min_layer); }
  // Compute the tributary layer
  for (auto& topo_struct : *topo_structs) { topo_struct->ComputeTributaryLayer(max_min_layer); }
}

void InitAllParameters(std::vector<TopoStruct*>* topo_structs,
                       HashMap<LogicalBlobId, int32_t>* lbi2id,
                       std::vector<std::vector<TopoStruct*>>* id2consumer_topo_structs,
                       std::vector<int64_t>* id2blob_size) {
  // Construct the map from a lbi to its id, consumers, blob size
  for (auto& topo_struct : *topo_structs) {
    const auto& consumer = topo_struct->op_node->op();
    for (const auto& ibn : consumer.input_bns()) {
      const LogicalBlobId& lbi = consumer.BnInOp2Lbi(ibn);
      auto it = lbi2id->find(lbi);
      if (it == lbi2id->end()) {
        (*lbi2id)[lbi] = id2blob_size->size();
        const BlobDesc& logical_blob_desc = topo_struct->op_node->LogicalBlobDesc4Lbi(lbi);
        id2blob_size->push_back(TotalByteSize4BlobDesc(logical_blob_desc));
        id2consumer_topo_structs->push_back({topo_struct});
      } else {
        id2consumer_topo_structs->at(it->second).push_back(topo_struct);
      }
    }
  }

  // Construct all the data edges and control edges
  InitInOutTopoStructs(topo_structs);

  // Compute the layers
  ComputeLayer(topo_structs);

  // Compute the memory increment for all the topological structures
  ComputeAllMemoryIncrement(*topo_structs, *lbi2id, *id2consumer_topo_structs, *id2blob_size);

  // Update sat, since sat might be changed in previous jobs
  UpdateSat(*topo_structs, &sat);

  // Decide which node should run first
  InitDecideParameters(sat, &decide_parameters);
  VLOG(3) << "Straightening order in sbp graph: ";
  for (int32_t decide_parameter : decide_parameters) { VLOG(3) << decide_parameter; }
}

void StraightenOpNodes(HashMap<const OpNode*, TopoStruct>& op_node2topo_struct,
                       std::vector<TopoStruct*>* topo_structs,
                       HashMap<LogicalBlobId, int32_t>* lbi2id,
                       std::vector<std::vector<TopoStruct*>>* id2consumer_topo_structs,
                       std::vector<int64_t>* id2blob_size,
                       std::vector<TopoStruct*>* ordered_topo_structs) {
  InitAllParameters(topo_structs, lbi2id, id2consumer_topo_structs, id2blob_size);

  std::set<TopoStruct*, comp> waiting_list;

  // Wait in the list
  auto wait = [&](TopoStruct* topo_struct) { waiting_list.insert(topo_struct); };

  // Initialization
  for (auto& topo_struct : *topo_structs) {
    topo_struct->counter = topo_struct->in_topo_structs.size();
    if (topo_struct->counter == 0) { wait(topo_struct); }
  }

  // Finish execution
  auto finish_execution = [&](TopoStruct* topo_struct) {
    for (auto& out : topo_struct->out_topo_structs) {
      out->counter--;
      if (out->counter == 0) { wait(out); }
    }
  };

  // Execute the first node in the waiting list
  // Make sure to check that waiting list is not empty before execution
  auto execute = [&]() {
    auto first_topo_struct = *waiting_list.begin();
    // Set the order of execution for sbp nodes
    ordered_topo_structs->push_back(first_topo_struct);
    waiting_list.erase(waiting_list.begin());
    finish_execution(first_topo_struct);
  };

  // straightening
  while (!waiting_list.empty()) { execute(); }
}

}  // anonymous namespace

// Use two function
void InitMemory(const OpGraph& op_graph, SbpGraph* sbp_graph, bool nccl_use_compute_stream) {
  // Generate topological data structure for each sbp node
  HashMap<const OpNode*, TopoStruct> op_node2topo_struct;
  std::vector<TopoStruct*> topo_structs;
  std::vector<TopoStruct*> ordered_topo_structs;

  // Traverse all the nodes in the sbp graph
  for (const auto& sbp_node : sbp_graph->GetNodeList()) {
    auto* op_node = sbp_node->GetOperatorNode();
    CHECK(op_node != nullptr)
        << "No proxy node allow at this status. InitMemory() should be run before sbp collector!";
    op_node2topo_struct.insert({op_node, TopoStruct(sbp_node)});
    topo_structs.push_back(&op_node2topo_struct.at(op_node));
  }

  // Construct the map from a lbi to its id, consumers, blob size
  HashMap<LogicalBlobId, int32_t> lbi2id;
  std::vector<std::vector<TopoStruct*>> id2consumer_topo_structs;
  std::vector<int64_t> id2blob_size;

  StraightenOpNodes(op_node2topo_struct, &topo_structs, &lbi2id, &id2consumer_topo_structs,
                    &id2blob_size, &ordered_topo_structs);

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

// Straighten a subset of the op graph
void StraightenSubGraph(const std::vector<const OpNode*>& sub_graph,
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

  StraightenOpNodes(op_node2topo_struct, &topo_structs, &lbi2id, &id2consumer_topo_structs,
                    &id2blob_size, &ordered_topo_structs);

  for (auto& ordered_topo_struct : ordered_topo_structs) {
    ordered_op_nodes->push_back(ordered_topo_struct->op_node);
  }
}

// Straighten the whole op graph
void StraightenOpGraph(const OpGraph& op_graph, std::vector<const OpNode*>* ordered_op_nodes) {
  std::vector<const OpNode*> sub_graph;

  // Traverse and store all the nodes in the op graph
  op_graph.ForEachNode([&](OpNode* node) { sub_graph.push_back(node); });

  StraightenSubGraph(sub_graph, ordered_op_nodes);
}

}  // namespace auto_parallel
}  // namespace oneflow
