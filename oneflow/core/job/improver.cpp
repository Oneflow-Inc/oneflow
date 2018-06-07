#include "oneflow/core/job/improver.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/register/register_desc.pb.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/profiler.h"

namespace oneflow {

namespace {

void AssertEnableMemSharingConsistency(const TaskProto* task_proto) {
  HashSet<bool> is_enable_mem_sharing;
  for (const auto& pair : task_proto->produced_regst_desc()) {
    if (pair.second.consumer_task_id_size() > 0
        && RtRegstDesc(pair.second).packed_blob_desc()->TotalByteSize() > 0) {
      is_enable_mem_sharing.insert(pair.second.enable_mem_sharing());
    }
  }
  CHECK_LE(is_enable_mem_sharing.size(), 1);
}

class MemSharedTaskNode;

class MemSharedTaskEdge final : public Edge<MemSharedTaskNode, MemSharedTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemSharedTaskEdge);
  MemSharedTaskEdge() = default;
  ~MemSharedTaskEdge() = default;
};

class MemSharedTaskNode final : public Node<MemSharedTaskNode, MemSharedTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemSharedTaskNode);
  explicit MemSharedTaskNode(const TaskProto& task_proto) : task_proto_(&task_proto) {}
  ~MemSharedTaskNode() = default;

  const TaskProto* task_proto() const { return task_proto_; }

 private:
  const TaskProto* task_proto_;
};

class MemSharedTaskGraph final : public Graph<const MemSharedTaskNode, MemSharedTaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemSharedTaskGraph);
  explicit MemSharedTaskGraph(const Plan& plan);
  ~MemSharedTaskGraph() = default;

  void ComputeLifetimeSameStreamActorIds(const RegstDescProto* regst_desc,
                                         HashSet<int64_t>* lifetime_same_stream_actor_ids) const;

 private:
  void InitNodes();
  void InitEdges();
  void InitNode2Ancestor();
  bool IsAnyOneReachable(const HashSet<const MemSharedTaskNode*>& nodes,
                         const MemSharedTaskNode* ancestor) const;

  const Plan* plan_;
  HashMap<int64_t, MemSharedTaskNode*> task_id2mem_shared_task_node_;
  HashMap<const MemSharedTaskNode*, HashSet<const MemSharedTaskNode*>> node2ancestor_;
};

MemSharedTaskGraph::MemSharedTaskGraph(const Plan& plan) : plan_(&plan) {
  InitNodes();
  InitEdges();
  InitNode2Ancestor();
}

void MemSharedTaskGraph::InitNodes() {
  for (const auto& task : plan_->task()) {
    MemSharedTaskNode* mem_shared_task_node = new MemSharedTaskNode(task);
    task_id2mem_shared_task_node_.insert({task.task_id(), mem_shared_task_node});
    AddAllocatedNode(mem_shared_task_node);
  }
}

void MemSharedTaskGraph::InitEdges() {
  for (const auto& task_id_and_mem_shared_task_node : task_id2mem_shared_task_node_) {
    MemSharedTaskNode* producer_node = task_id_and_mem_shared_task_node.second;
    AssertEnableMemSharingConsistency(producer_node->task_proto());
    for (const auto& pair : producer_node->task_proto()->produced_regst_desc()) {
      if (pair.second.enable_mem_sharing()
          || RtRegstDesc(pair.second).packed_blob_desc()->TotalByteSize() == 0) {
        for (int64_t consumer_task_id : pair.second.consumer_task_id()) {
          Connect(producer_node, NewEdge(), task_id2mem_shared_task_node_.at(consumer_task_id));
        }
      }
    }
  }
}

void MemSharedTaskGraph::InitNode2Ancestor() {
  TopoForEachNode([&](const MemSharedTaskNode* node) {
    node->ForEachNodeOnInEdge([&](const MemSharedTaskNode* prev) {
      node2ancestor_[node].insert(prev);
      node2ancestor_[node].insert(node2ancestor_[prev].begin(), node2ancestor_[prev].end());
    });
  });
}

bool MemSharedTaskGraph::IsAnyOneReachable(const HashSet<const MemSharedTaskNode*>& nodes,
                                           const MemSharedTaskNode* ancestor) const {
  for (const MemSharedTaskNode* node : nodes) {
    if (node2ancestor_.at(node).find(ancestor) != node2ancestor_.at(node).end()) { return true; }
  }
  return false;
}

void MemSharedTaskGraph::ComputeLifetimeSameStreamActorIds(
    const RegstDescProto* regst_desc, HashSet<int64_t>* lifetime_same_stream_actor_ids) const {
  const auto* producer = task_id2mem_shared_task_node_.at(regst_desc->producer_task_id());
  HashSet<const MemSharedTaskNode*> consumers;
  for (int64_t consumer_task_id : regst_desc->consumer_task_id()) {
    consumers.insert(task_id2mem_shared_task_node_.at(consumer_task_id));
  }
  auto ForEachInNode = [&](const MemSharedTaskNode* node,
                           const std::function<void(const MemSharedTaskNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](const MemSharedTaskNode* prev) {
      if (prev == producer || IsAnyOneReachable({prev}, producer)) { Handler(prev); }
    });
  };
  auto ForEachOutNode = [&](const MemSharedTaskNode* node,
                            const std::function<void(const MemSharedTaskNode*)>& Handler) {
    node->ForEachNodeOnOutEdge([&](const MemSharedTaskNode* next) {
      if (consumers.find(next) != consumers.end() || IsAnyOneReachable(consumers, next)) {
        Handler(next);
      }
    });
  };
  int64_t global_work_stream_id =
      Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(regst_desc->producer_task_id());
  TopoForEachNode({producer}, ForEachInNode, ForEachOutNode, [&](const MemSharedTaskNode* node) {
    int64_t task_id = node->task_proto()->task_id();
    if (Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(task_id) == global_work_stream_id) {
      lifetime_same_stream_actor_ids->insert(task_id);
    }
  });
}

class RegstLifetimePosetNode;

class RegstLifetimePosetEdge final : public Edge<RegstLifetimePosetNode, RegstLifetimePosetEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstLifetimePosetEdge);
  RegstLifetimePosetEdge() = default;
  ~RegstLifetimePosetEdge() = default;
};

class RegstLifetimePosetNode final : public Node<RegstLifetimePosetNode, RegstLifetimePosetEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstLifetimePosetNode);
  RegstLifetimePosetNode(const RegstDescProto* regst_desc,
                         std::unique_ptr<HashSet<int64_t>>&& lifetime_same_stream_actor_ids)
      : regst_desc_(regst_desc),
        lifetime_same_stream_actor_ids_(std::move(lifetime_same_stream_actor_ids)) {}
  ~RegstLifetimePosetNode() = default;

  const RegstDescProto& regst_desc() const { return *regst_desc_; }
  const HashSet<int64_t>& lifetime_same_stream_actor_ids() const {
    return *lifetime_same_stream_actor_ids_;
  }

 private:
  const RegstDescProto* regst_desc_;
  std::unique_ptr<HashSet<int64_t>> lifetime_same_stream_actor_ids_;
};

class RegstLifetimePosetGraph final
    : public Graph<const RegstLifetimePosetNode, RegstLifetimePosetEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstLifetimePosetGraph);
  RegstLifetimePosetGraph(const std::list<const RegstDescProto*>& regst_descs,
                          const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>&
                              ComputeLifetimeSameStreamActorIds);
  ~RegstLifetimePosetGraph() = default;

  void ForEachLayerwiseSameColoredRegstDescIds(
      const std::function<void(const std::list<int64_t>&)>&) const;

 private:
  void InitNodesAndEdges(const std::list<const RegstDescProto*>& regst_descs,
                         const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>&
                             ComputeLifetimeSameStreamActorIds);
  void InitRegstDesc2IntersectedRegstDescs();
  bool LifetimeContain(const RegstLifetimePosetNode* long_lifetime_node,
                       const RegstLifetimePosetNode* short_lifetime_node) const;
  void ForEachSameColoredRegstDescIds(
      const HashSet<const RegstLifetimePosetNode*>& layer_nodes,
      const std::function<void(const std::list<int64_t>&)>& Handler) const;

  HashMap<const RegstLifetimePosetNode*, HashSet<const RegstLifetimePosetNode*>>
      regst_lifetime_node2intersected_nodes_;
};

RegstLifetimePosetGraph::RegstLifetimePosetGraph(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>&
        ComputeLifetimeSameStreamActorIds) {
  InitNodesAndEdges(regst_descs, ComputeLifetimeSameStreamActorIds);
  InitRegstDesc2IntersectedRegstDescs();
}

void RegstLifetimePosetGraph::InitNodesAndEdges(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>&
        ComputeLifetimeSameStreamActorIds) {
  // init nodes
  std::list<RegstLifetimePosetNode*> nodes;
  for (const RegstDescProto* regst_desc : regst_descs) {
    auto lifetime_same_stream_actor_ids = std::make_unique<HashSet<int64_t>>();
    ComputeLifetimeSameStreamActorIds(regst_desc, lifetime_same_stream_actor_ids.get());
    auto* node = new RegstLifetimePosetNode(regst_desc, std::move(lifetime_same_stream_actor_ids));
    AddAllocatedNode(node);
    nodes.push_back(node);
  }
  // init edges
  HashMap<const RegstLifetimePosetNode*, size_t> node2size;
  for (const auto* node : nodes) {
    node2size[node] = RtRegstDesc(node->regst_desc()).packed_blob_desc()->TotalByteSize();
  }
  for (RegstLifetimePosetNode* src : nodes) {
    for (RegstLifetimePosetNode* dst : nodes) {
      if (src == dst) { break; }
      if (LifetimeContain(dst, src)
          && (!LifetimeContain(src, dst) || node2size.at(src) > node2size.at(dst))) {
        Connect(src, NewEdge(), dst);
      }
    }
  }
}

bool RegstLifetimePosetGraph::LifetimeContain(
    const RegstLifetimePosetNode* long_lifetime_node,
    const RegstLifetimePosetNode* short_lifetime_node) const {
  for (int64_t actor_id : short_lifetime_node->lifetime_same_stream_actor_ids()) {
    if (long_lifetime_node->lifetime_same_stream_actor_ids().find(actor_id)
        == long_lifetime_node->lifetime_same_stream_actor_ids().end()) {
      return false;
    }
  }
  return true;
}

void RegstLifetimePosetGraph::InitRegstDesc2IntersectedRegstDescs() {
  HashMap<int64_t, HashSet<const RegstLifetimePosetNode*>> actor_id2node;
  ForEachNode([&](const RegstLifetimePosetNode* node) {
    for (int64_t actor_id : node->lifetime_same_stream_actor_ids()) {
      actor_id2node[actor_id].insert(node);
    }
  });
  for (const auto& pair : actor_id2node) {
    for (const RegstLifetimePosetNode* node : pair.second) {
      regst_lifetime_node2intersected_nodes_[node].insert(pair.second.begin(), pair.second.end());
    }
  }
  for (auto& pair : regst_lifetime_node2intersected_nodes_) { pair.second.erase(pair.first); }
}

void RegstLifetimePosetGraph::ForEachSameColoredRegstDescIds(
    const HashSet<const RegstLifetimePosetNode*>& layer_nodes,
    const std::function<void(const std::list<int64_t>&)>& Handler) const {
  auto ForEachIntersected = [&](const RegstLifetimePosetNode* node,
                                const std::function<void(const RegstLifetimePosetNode*)>& Handler) {
    for (auto* intersected_node : regst_lifetime_node2intersected_nodes_.at(node)) {
      if (layer_nodes.find(intersected_node) != layer_nodes.end()) { Handler(intersected_node); }
    }
  };
  HashMap<const RegstLifetimePosetNode*, std::set<int32_t>> node2excluded_color_ids;
  HashMap<const RegstLifetimePosetNode*, int32_t> node2color_id;
  for (const RegstLifetimePosetNode* start : layer_nodes) {
    if (node2color_id.find(start) != node2color_id.end()) { continue; }
    BfsForEachNode({start}, ForEachIntersected, [&](const RegstLifetimePosetNode* node) {
      if (node2color_id.find(node) != node2color_id.end()) { return; }
      int32_t color_id = 0;
      const auto& excluded_color_ids = node2excluded_color_ids.at(node);
      for (; excluded_color_ids.find(color_id) != excluded_color_ids.end(); ++color_id) {}
      node2color_id[node] = color_id;
      ForEachIntersected(node, [&](const RegstLifetimePosetNode* intersected) {
        if (node2color_id.find(intersected) != node2color_id.end()) { return; }
        node2excluded_color_ids[intersected].insert(color_id);
      });
    });
  }
  HashMap<int32_t, std::list<int64_t>> color_id2regst_desc_ids;
  for (const auto& pair : node2color_id) {
    color_id2regst_desc_ids[pair.second].push_back(pair.first->regst_desc().regst_desc_id());
  }
  for (const auto& pair : color_id2regst_desc_ids) { Handler(pair.second); }
}

void RegstLifetimePosetGraph::ForEachLayerwiseSameColoredRegstDescIds(
    const std::function<void(const std::list<int64_t>&)>& Handler) const {
  HashSet<const RegstLifetimePosetNode*> handled_nodes;
  auto GetInNodesNum = [&](const RegstLifetimePosetNode* node) -> size_t {
    size_t num = 0;
    node->ForEachNodeOnInEdge([&](const RegstLifetimePosetNode* in_node) {
      if (handled_nodes.find(in_node) == handled_nodes.end()) { ++num; }
    });
    return num;
  };
  while (true) {
    HashSet<const RegstLifetimePosetNode*> cur_layer_nodes;
    ForEachNode([&](const RegstLifetimePosetNode* node) {
      if (GetInNodesNum(node) == 0) { cur_layer_nodes.insert(node); }
    });
    if (cur_layer_nodes.empty()) { break; }
    ForEachSameColoredRegstDescIds(cur_layer_nodes, Handler);
    handled_nodes.insert(cur_layer_nodes.begin(), cur_layer_nodes.end());
  }
}

void ForEachComputeStreamRegstDescs(
    const Plan& plan, const std::function<void(const std::list<const RegstDescProto*>&)>& Handler) {
  HashMap<int64_t, std::list<const RegstDescProto*>> global_work_stream_id2regst_descs;
  for (const auto& task : plan.task()) {
    if (Global<IDMgr>::Get()->LocalWorkStreamId4TaskId(task.task_id()) == 0) {
      int64_t global_work_stream_id =
          Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(task.task_id());
      for (const auto& pair : task.produced_regst_desc()) {
        global_work_stream_id2regst_descs[global_work_stream_id].push_back(&pair.second);
      }
    }
  }
  for (const auto& pair : global_work_stream_id2regst_descs) { Handler(pair.second); }
}

bool IsConsumersAndProducerAllInComputeStream(const RegstDescProto* regst_desc) {
  HashSet<int64_t> stream_ids;
  int64_t producer_task_id = regst_desc->producer_task_id();
  stream_ids.insert(Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(producer_task_id));
  for (int64_t consumer_task_id : regst_desc->consumer_task_id()) {
    stream_ids.insert(Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(consumer_task_id));
  }
  return stream_ids.size() == 1
         && Global<IDMgr>::Get()->LocalWorkStreamId4TaskId(producer_task_id) == 0;
}

std::list<const RegstDescProto*> SelectSharableRegstDescsWithConsumer(
    const std::list<const RegstDescProto*>& regst_descs) {
  std::list<const RegstDescProto*> sharable_regst_descs_with_consumer;
  for (const RegstDescProto* regst_desc : regst_descs) {
    if (regst_desc->consumer_task_id_size() > 0 && regst_desc->enable_mem_sharing()
        && regst_desc->register_num() == 1
        && IsConsumersAndProducerAllInComputeStream(regst_desc)) {
      sharable_regst_descs_with_consumer.push_back(regst_desc);
    }
  }
  return sharable_regst_descs_with_consumer;
}

std::list<const RegstDescProto*> SelectRegstDescsWithoutConsumer(
    const std::list<const RegstDescProto*>& regst_descs) {
  std::list<const RegstDescProto*> regst_descs_without_consumer;
  for (const RegstDescProto* regst_desc : regst_descs) {
    if (regst_desc->consumer_task_id_size() == 0) {
      CHECK(regst_desc->enable_mem_sharing());
      regst_descs_without_consumer.push_back(regst_desc);
    }
  }
  return regst_descs_without_consumer;
}

void ForEachImprovedMemSharedId(const Plan& plan,
                                const std::function<void(int64_t, int64_t)>& Handler) {
  MemSharedTaskGraph mem_shared_grph(plan);
  auto ComputeLifetimeSameStreamActorIds = [&](const RegstDescProto* regst_desc,
                                               HashSet<int64_t>* lifetime_same_stream_actor_ids) {
    CHECK(regst_desc->enable_mem_sharing());
    mem_shared_grph.ComputeLifetimeSameStreamActorIds(regst_desc, lifetime_same_stream_actor_ids);
  };
  ForEachComputeStreamRegstDescs(plan, [&](const std::list<const RegstDescProto*>& regst_descs) {
    const auto& regst_descs_without_consumer = SelectRegstDescsWithoutConsumer(regst_descs);
    const auto& sharable_with_consumer = SelectSharableRegstDescsWithConsumer(regst_descs);
    int mem_shared_id = 0;
    auto AllocateMemSharedId = [&](const std::list<int64_t>& regst_desc_ids) {
      for (int64_t regst_desc_id : regst_desc_ids) { Handler(regst_desc_id, mem_shared_id); }
      ++mem_shared_id;
    };
    RegstLifetimePosetGraph(regst_descs_without_consumer, ComputeLifetimeSameStreamActorIds)
        .ForEachLayerwiseSameColoredRegstDescIds(AllocateMemSharedId);
    RegstLifetimePosetGraph(sharable_with_consumer, ComputeLifetimeSameStreamActorIds)
        .ForEachLayerwiseSameColoredRegstDescIds(AllocateMemSharedId);
  });
}

double CalcRegstNum(double regst_desc_duration, double ii, double ii_scale) {
  return ((ii_scale - 1) * ii + regst_desc_duration) / (ii_scale * ii);
}

double CalcII(double regst_desc_duration, uint64_t regst_num, double ii_scale) {
  return regst_desc_duration / ((regst_num - 1) * ii_scale + 1);
}

uint64_t CalcRegstNum(
    const RegstDescProto& regst_desc,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    double ii,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId) {
  int64_t regst_desc_id = regst_desc.regst_desc_id();
  const auto& consumer_actor_id2duration = PathDurations4RegstDescId(regst_desc_id);
  const auto& consumer_actor_id2ii_scale = PathIIScales4RegstDescId(regst_desc_id);
  uint64_t regst_num = 0;
  for (const auto& pair : consumer_actor_id2duration) {
    double duration = pair.second;
    double ii_scale = consumer_actor_id2ii_scale.at(pair.first);
    uint64_t cur_path_regst_num = ceil(CalcRegstNum(duration, ii, ii_scale));
    regst_num = std::max(regst_num, cur_path_regst_num);
  }
  regst_num = std::max(regst_num, static_cast<uint64_t>(regst_desc.min_register_num()));
  regst_num = std::min(regst_num, static_cast<uint64_t>(regst_desc.max_register_num()));
  return regst_num;
}

void ParseActEvents(const std::string& act_event_filepath, std::list<ActEvent>* act_events) {
  NormalPersistentInStream in_stream(LocalFS(), act_event_filepath);
  int64_t act_event_size;
  while (!in_stream.Read(reinterpret_cast<char*>(&act_event_size), sizeof(act_event_size))) {
    std::vector<char> buffer(act_event_size);
    CHECK(!in_stream.Read(buffer.data(), act_event_size));
    act_events->emplace_back();
    act_events->back().ParseFromArray(buffer.data(), act_event_size);
  }
}

uint64_t CalcMemoryConsumed(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
    double ii) {
  uint64_t mem_consuming = 0;
  HashMap<int64_t, uint64_t> mem_shared_id2max_regst_desc_mem_bytes;
  for (const RegstDescProto* regst_desc : regst_descs) {
    uint64_t regst_num =
        CalcRegstNum(*regst_desc, PathDurations4RegstDescId, ii, PathIIScales4RegstDescId);
    uint64_t total_byte_size = RtRegstDesc(*regst_desc).packed_blob_desc()->TotalByteSize();
    if (regst_desc->mem_shared_id() == -1) {
      mem_consuming += regst_num * total_byte_size;
    } else {
      CHECK(!regst_desc->enable_mem_sharing());
      CHECK_EQ(regst_num, 1);
      auto& max_bytes = mem_shared_id2max_regst_desc_mem_bytes[regst_desc->mem_shared_id()];
      max_bytes = std::max(max_bytes, total_byte_size);
    }
  }
  for (const auto& pair : mem_shared_id2max_regst_desc_mem_bytes) { mem_consuming += pair.second; }
  return mem_consuming;
}

std::shared_ptr<HashMap<int64_t, RegstDescProto*>> MakeRegstDescId2RegstDesc(Plan* plan) {
  auto regst_desc_id2regst_desc = std::make_shared<HashMap<int64_t, RegstDescProto*>>();
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      int64_t regst_desc_id = pair.second.regst_desc_id();
      regst_desc_id2regst_desc->insert({regst_desc_id, &pair.second});
    }
  }
  return regst_desc_id2regst_desc;
}

std::function<void(int64_t, uint64_t)> MakeSetterSetPlanRegstNum(Plan* plan) {
  auto regst_desc_id2regst_desc = MakeRegstDescId2RegstDesc(plan);
  return [regst_desc_id2regst_desc](int64_t regst_desc_id, uint64_t num) {
    regst_desc_id2regst_desc->at(regst_desc_id)->set_register_num(num);
  };
}

std::function<void(int64_t, int64_t)> MakeSetterSetPlanMemSharedId(Plan* plan) {
  auto regst_desc_id2regst_desc = MakeRegstDescId2RegstDesc(plan);
  return [regst_desc_id2regst_desc](int64_t regst_desc_id, int64_t mem_shared_id) {
    regst_desc_id2regst_desc->at(regst_desc_id)->set_mem_shared_id(mem_shared_id);
  };
}

std::function<const HashMap<int64_t, double>&(int64_t)> MakeGetterPathDurations4RegstDescId(
    const ActGraph& graph) {
  auto regst_desc_id2consumer_id2duration =
      std::make_shared<HashMap<int64_t, HashMap<int64_t, double>>>();
  graph.ForEachRegstDescConsumerPathMeanDuration(
      [&](int64_t regst_desc_id, int64_t consumer_actor_id, double time) {
        (*regst_desc_id2consumer_id2duration)[regst_desc_id][consumer_actor_id] = time;
      });
  auto empty = std::make_shared<const HashMap<int64_t, double>>();
  return [regst_desc_id2consumer_id2duration,
          empty](int64_t regst_desc_id) -> const HashMap<int64_t, double>& {
    const auto& it = regst_desc_id2consumer_id2duration->find(regst_desc_id);
    if (it == regst_desc_id2consumer_id2duration->end()) {
      return *empty;
    } else {
      return it->second;
    }
  };
}

uint64_t NumOfPiecesInSnapshot() {
  return Global<JobDesc>::Get()->NumOfBatchesInSnapshot()
         * Global<JobDesc>::Get()->NumOfPiecesInBatch();
}

double FormalDuration4ExperimentalDuration(TaskType task_type, double duration,
                                           double act_frequency) {
  if (task_type == TaskType::kMdSave) {
    double formal_run_frequency = 1.0 / NumOfPiecesInSnapshot();
    return (duration / act_frequency) * formal_run_frequency;
  }
  return duration;
}

double CalcBaseII(const ActGraph& act_graph) {
  int64_t max_act_cnt = 0;
  for (const auto& pair : act_graph.actor_id2act_cnt()) {
    if (max_act_cnt < pair.second) { max_act_cnt = pair.second; }
  }
  HashMap<int64_t, double> actor_id2act_frequency;
  for (const auto& pair : act_graph.actor_id2act_cnt()) {
    actor_id2act_frequency[pair.first] = 1.0 * pair.second / max_act_cnt;
  }
  HashMap<int64_t, double> stream_id2total_calc_time;
  act_graph.ForEachNode([&](const ActNode* act_node) {
    int64_t stream_id = act_node->act_event().work_stream_id();
    int64_t actor_id = act_node->actor_id();
    TaskType task_type = act_graph.GetTaskProto(actor_id).task_type();
    stream_id2total_calc_time[stream_id] += FormalDuration4ExperimentalDuration(
        task_type, act_node->Duration(), actor_id2act_frequency.at(actor_id));
  });
  double base_ii = 0;
  for (const auto& pair : stream_id2total_calc_time) {
    base_ii = std::max(base_ii, pair.second / max_act_cnt);
  }
  return base_ii;
}

double IIScale4Actor(TaskType task_type, double default_ii_scale) {
  if (task_type == TaskType::kMdSave) { return NumOfPiecesInSnapshot(); }
  return default_ii_scale;
}

void PushAvgActTimeToProfiler(const ActGraph& act_graph) {
  for (const auto& pair : act_graph.actor_id2total_act_time()) {
    double act_time = pair.second / act_graph.actor_id2act_cnt().at(pair.first);
    Global<Profiler>::Get()->PushAvgActTime(pair.first, act_time);
  }
}

std::function<const HashMap<int64_t, double>&(int64_t)> MakeGetterPathIIScales4RegstDescId(
    const ActGraph& graph) {
  auto regst_desc_id2consumer_id2ii_scale =
      std::make_shared<HashMap<int64_t, HashMap<int64_t, double>>>();
  graph.ForEachRegstDescConsumerPathIIScale(
      [&](int64_t regst_desc_id, int64_t consumer_actor_id, double ii_scale) {
        TaskType task_type = graph.GetTaskProto(consumer_actor_id).task_type();
        (*regst_desc_id2consumer_id2ii_scale)[regst_desc_id][consumer_actor_id] =
            IIScale4Actor(task_type, ii_scale);
      });
  auto empty = std::make_shared<const HashMap<int64_t, double>>();
  return [regst_desc_id2consumer_id2ii_scale,
          empty](int64_t regst_desc_id) -> const HashMap<int64_t, double>& {
    const auto& it = regst_desc_id2consumer_id2ii_scale->find(regst_desc_id);
    if (it == regst_desc_id2consumer_id2ii_scale->end()) {
      return *empty;
    } else {
      return it->second;
    }
  };
}

}  // namespace

uint64_t Improver::AvailableMemSize(int64_t machine_id, int64_t memory_zone_id) const {
  int64_t mem_size = amd_.machine_amd(machine_id).zone_size(memory_zone_id);
  JobDesc* job_desc = Global<JobDesc>::Get();
  if (memory_zone_id == job_desc->GpuDeviceNum()) {
    mem_size -= job_desc->reserved_host_mem_byte();
    mem_size -= job_desc->persistence_buf_byte() * record_load_task_num_.at(machine_id);
  } else {
    mem_size -= job_desc->reserved_device_mem_byte();
  }
  CHECK_GT(mem_size, 0);
  return static_cast<uint64_t>(mem_size);
}

int64_t Improver::GetMemoryZoneId(const MemoryCase& mem_case) const {
  if (mem_case.has_device_cuda_mem()) {
    return mem_case.device_cuda_mem().device_id();
  } else {
    return Global<JobDesc>::Get()->GpuDeviceNum();
  }
}

void Improver::MakeMemZoneRegstDescs(const Plan& plan, MemZoneRegstDescs* mz2regst_desc) const {
  mz2regst_desc->resize(amd_.machine_amd_size());
  FOR_RANGE(int64_t, machine_id, 0, amd_.machine_amd_size()) {
    mz2regst_desc->at(machine_id).resize(amd_.machine_amd(machine_id).zone_size_size());
  }
  for (const auto& task : plan.task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      int64_t mem_zone_id = GetMemoryZoneId(pair.second.mem_case());
      mz2regst_desc->at(task.machine_id()).at(mem_zone_id).push_back(&pair.second);
    }
  }
}

bool Improver::IsAnyZoneOutOfMemory(
    const MemZoneRegstDescs& mz_regst_descs,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
    double ii) const {
  FOR_RANGE(int64_t, machine_id, 0, mz_regst_descs.size()) {
    FOR_RANGE(int64_t, mem_zone_id, 0, mz_regst_descs[machine_id].size()) {
      const auto& regst_descs = mz_regst_descs[machine_id][mem_zone_id];
      if (CalcMemoryConsumed(regst_descs, PathDurations4RegstDescId, PathIIScales4RegstDescId, ii)
          >= AvailableMemSize(machine_id, mem_zone_id)) {
        return true;
      }
    }
  }
  return false;
}

double Improver::CalcMaxRegstDescDuration(
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const MemZoneRegstDescs& mz_regst_descs) const {
  double max_duration = 0;
  for (const auto& zone_regst_descs : mz_regst_descs) {
    for (const auto& regst_descs : zone_regst_descs) {
      for (const RegstDescProto* regst_desc : regst_descs) {
        for (const auto& pair : PathDurations4RegstDescId(regst_desc->regst_desc_id())) {
          max_duration = std::max(max_duration, pair.second);
        }
      }
    }
  }
  return max_duration;
}

double Improver::BinarySearchII(
    double base_ii,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
    const MemZoneRegstDescs& mz_regst_descs) const {
  double max_duration = CalcMaxRegstDescDuration(PathDurations4RegstDescId, mz_regst_descs);
  CHECK(!IsAnyZoneOutOfMemory(mz_regst_descs, PathDurations4RegstDescId, PathIIScales4RegstDescId,
                              max_duration));
  const double ii_search_threshold = 1;
  double r = max_duration;
  double l = base_ii;
  double mid = base_ii;
  while ((r - l) > ii_search_threshold) {
    mid = (l + r) / 2;
    if (IsAnyZoneOutOfMemory(mz_regst_descs, PathDurations4RegstDescId, PathIIScales4RegstDescId,
                             mid)) {
      l = mid;
    } else {
      r = mid;
    }
  }
  return r;
}

void Improver::ForEachImprovedRegstNum(
    const ActGraph& graph, const Plan& plan, bool is_memory_limited,
    const std::function<void(int64_t, uint64_t)>& Handler) const {
  auto PathDurations4RegstDescId = MakeGetterPathDurations4RegstDescId(graph);
  auto PathIIScales4RegstDescId = MakeGetterPathIIScales4RegstDescId(graph);
  double ii = CalcBaseII(graph);
  if (is_memory_limited) {
    MemZoneRegstDescs mz_regst_descs;
    MakeMemZoneRegstDescs(plan, &mz_regst_descs);
    ii = BinarySearchII(ii, PathDurations4RegstDescId, PathIIScales4RegstDescId, mz_regst_descs);
  }
  LOG(INFO) << "memory " << (is_memory_limited ? "limited" : "unlimited") << " ii: " << ii;
  for (const auto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      uint64_t regst_num =
          CalcRegstNum(pair.second, PathDurations4RegstDescId, ii, PathIIScales4RegstDescId);
      Handler(pair.second.regst_desc_id(), regst_num);
    }
  }
}

Plan Improver::Improve(const AvailableMemDesc& amd, const Plan& naive_plan,
                       const std::string& act_event_filepath) {
  amd_ = amd;
  record_load_task_num_.assign(Global<JobDesc>::Get()->TotalMachineNum(), 0);
  for (const TaskProto& task_proto : naive_plan.task()) {
    if (task_proto.task_type() == TaskType::kRecordLoad) {
      record_load_task_num_.at(Global<IDMgr>::Get()->MachineId4ActorId(task_proto.task_id())) += 1;
    }
  }
  auto act_events = std::make_unique<std::list<ActEvent>>();
  ParseActEvents(act_event_filepath, act_events.get());
  ActGraph act_graph(naive_plan, std::move(act_events));
  PushAvgActTimeToProfiler(act_graph);
  Plan mem_unlimited_plan(naive_plan);
  ForEachImprovedRegstNum(act_graph, naive_plan, false,
                          MakeSetterSetPlanRegstNum(&mem_unlimited_plan));
  Plan mem_shared_plan(mem_unlimited_plan);
  ForEachImprovedMemSharedId(mem_unlimited_plan, MakeSetterSetPlanMemSharedId(&mem_shared_plan));
  Plan plan(mem_shared_plan);
  ForEachImprovedRegstNum(act_graph, mem_shared_plan, true, MakeSetterSetPlanRegstNum(&plan));
  return plan;
}

Plan Improver::ImproveMemSharedIdOnly(const Plan& naive_plan) const {
  Plan plan(naive_plan);
  ForEachImprovedMemSharedId(naive_plan, MakeSetterSetPlanMemSharedId(&plan));
  return plan;
}

}  // namespace oneflow
