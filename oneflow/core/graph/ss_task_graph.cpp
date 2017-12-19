#include "oneflow/core/graph/ss_task_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/trial_run/utilization_manager.h"

namespace oneflow {

namespace {

void ForEachEdgeFromPlan(
    const Plan& plan,
    const std::function<void(uint64_t, uint64_t)>& DoEachEdge) {
  std::unordered_map<int64_t, const TaskProto*> id2task_proto;
  for (const TaskProto& task_proto : plan.task()) {
    id2task_proto[task_proto.id()] = &task_proto;
  }
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      for (int64_t consumer_id : pair.second.consumer_task_id()) {
        //  avoid circle
        if (task_proto.type() == kMdUpdtCompTask
            && id2task_proto[consumer_id]->type() == kDataCompTask) {
          continue;
        }
        DoEachEdge(task_proto.id(), consumer_id);
      }
    }
  }
}

void GetUtilizationEventPackages(
    const std::string& event_package_dir, fs::FileSystem* fs,
    std::list<UtilizationEventPackageProto>* event_packages) {
  for (const std::string& filepath : fs->ListDir(event_package_dir)) {
    UtilizationEventPackageProto evt_pkg;
    event_packages->push_back(evt_pkg);
    NormalPersistentInStream in_stream(fs,
                                       JoinPath(event_package_dir, filepath));
    std::string log_file;
    in_stream.ReadAll(&log_file);
    ParseProtoFromString(log_file, &event_packages->back());
  }
}

float ConsumingTimePerPiece(
    const std::list<const UtilizationProto*>& utili_protos) {
  std::unordered_set<int64_t> piece_ids;
  float total_consumming = 0;
  for (const auto* utilization : utili_protos) {
    total_consumming += utilization->end_at() - utilization->start_at();
    piece_ids.insert(utilization->start_piece_id());
  }
  return total_consumming / piece_ids.size();
}

double PackageLastEventEndAt(const UtilizationEventPackageProto& package) {
  double last_end_at = 0.0;
  for (int i = 0; i < package.event_size(); ++i) {
    last_end_at = std::max(last_end_at, package.event(i).time());
  }
  return last_end_at;
}

int64_t PackageLastPieceId(const UtilizationEventPackageProto& package) {
  int64_t last_piece_id = 0.0;
  for (int i = 0; i < package.event_size(); ++i) {
    last_piece_id = std::max(last_piece_id, package.event(i).piece_id());
  }
  return last_piece_id;
}

}  // namespace

std::string SSTaskNode::GetDeviceName() const {
  uint64_t thrd_loc_id =
      IDMgr::Singleton()->ThrdLocId4ActorId(task_proto().id());
  std::string thrd_loc_name = "";
  if (thrd_loc_id == IDMgr::Singleton()->PersistenceThrdLocId()) {
    thrd_loc_name = "persistence";
  } else if (thrd_loc_id == IDMgr::Singleton()->BoxingThrdLocId()) {
    thrd_loc_name = "boxing";
  } else if (thrd_loc_id == IDMgr::Singleton()->CommNetThrdLocId()) {
    thrd_loc_name = "comm_net";
  } else {
    thrd_loc_name = std::to_string(thrd_loc_id);
    if (task_proto().type() == TaskType::kCopyHdTask) {
      thrd_loc_name = thrd_loc_name + ":copy_hd";
    }
  }
  return IDMgr::Singleton()->MachineName4MachineId(
             IDMgr::Singleton()->MachineId4ActorId(task_proto().id()))
         + ":" + thrd_loc_name;
}

std::string SSTaskNode::VisualStr() const {
  std::string name = std::to_string(task_proto_->id()) + "\\n";
  if (task_proto_->type() != TaskType::kDataCompTask) {
    name = name + TaskType_Name(task_proto_->type());
  } else {
    for (int i = 0; i < task_proto_->exec_sequence().exec_node_size(); ++i) {
      name = name + task_proto_->exec_sequence().exec_node(i).op_name() + "\\n";
    }
  }
  return name;
}

std::string SSTaskNode::GlobalUniqueStreamName(uint64_t stream_id) const {
  uint64_t machine_id = task_proto().machine_id();
  return std::to_string(machine_id) + ":" + std::to_string(stream_id);
}

void SSTaskGraph::ForEachRegstDesc(
    const std::function<void(const RegstDescProto&)>& DoEach) const {
  for (const TaskProto& task_proto : plan().task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      DoEach(pair.second);
    }
  }
}

bool SSTaskGraph::IsAncestor(const SSTaskNode* asc,
                             const SSTaskNode* node) const {
  return task2ancestors_.at(node).find(asc) != task2ancestors_.at(node).end();
}

void SSTaskGraph::MakeRegstDescId2AvgLifeTimeHash(
    std::unordered_map<uint64_t, float>* regst_desc_id2life_time,
    const std::function<float(uint64_t)>& AvgDuration4TaskId) const {
  auto ForEachNext = [&](const SSTaskNode* task,
                         const std::function<void(const SSTaskNode*)>& DoEach) {
    for (SSTaskEdge* edge : task->out_edges()) { DoEach(edge->dst_node()); }
  };
  auto ForEachPrev = [&](const SSTaskNode* task,
                         const std::function<void(const SSTaskNode*)>& DoEach) {
    for (SSTaskEdge* edge : task->in_edges()) { DoEach(edge->src_node()); }
  };
  auto IsAscendant = std::bind(&SSTaskGraph::IsAncestor, this,
                               std::placeholders::_1, std::placeholders::_2);
  LongestPathVisitor<const SSTaskNode*> lpath_visitor(ForEachNext, ForEachPrev,
                                                      IsAscendant);
  auto AvgDuration4Task = [&](const SSTaskNode* task) -> double {
    return AvgDuration4TaskId(task->task_id());
  };
  ForEachRegstDesc([&](const RegstDescProto& regst_desc) {
    float life_time = 0;
    auto producer = GetSSTaskNode(regst_desc.producer_task_id());
    CHECK(producer);
    for (int64_t consumer_id : regst_desc.consumer_task_id()) {
      auto consumer = GetSSTaskNode(consumer_id);
      life_time = std::max(life_time, AvgLifeTime(lpath_visitor, producer,
                                                  consumer, AvgDuration4Task));
    }
    regst_desc_id2life_time->emplace(regst_desc.regst_desc_id(), life_time);
  });
}

float SSTaskGraph::AvgLifeTime(
    const LongestPathVisitor<const SSTaskNode*>& lpath_visitor,
    const SSTaskNode* start_task, const SSTaskNode* end_task,
    const std::function<double(const SSTaskNode* task)>& AvgDuration4Task)
    const {
  float life_time = 0;
  lpath_visitor(start_task, end_task, AvgDuration4Task,
                [&](const std::list<const SSTaskNode*>& path) {
                  if (path.back() == end_task) {
                    float d = 0;
                    for (auto task : path) { d += AvgDuration4Task(task); }
                    life_time = std::max(life_time, d);
                  }
                });
  return life_time;
}

void SSTaskGraph::UpdateAncestors() {
  ConstTopoForEachNode([&](const SSTaskNode* task) {
    for (const SSTaskEdge* edge : task->in_edges()) {
      const SSTaskNode* in_task = edge->src_node();
      for (const SSTaskNode* ancestor : task2ancestors_[in_task]) {
        task2ancestors_[task].insert(ancestor);
      }
      task2ancestors_[task].insert(in_task);
    }
  });
}

bool SSTaskGraph::ReachableWithoutEdge(const SSTaskEdge* edge) const {
  for (const SSTaskEdge* in_edge_of_dst : edge->dst_node()->in_edges()) {
    const SSTaskNode* in_node_of_dst = in_edge_of_dst->src_node();
    if (task2ancestors_.at(in_node_of_dst).find(edge->src_node())
        != task2ancestors_.at(in_node_of_dst).end()) {
      return true;
    }
  }
  return false;
}

void SSTaskGraph::RemoveMeaninglessEdges() {
  std::unordered_set<SSTaskEdge*> useless_edges;
  ForEachEdge([&](SSTaskEdge* edge) {
    if (ReachableWithoutEdge(edge)) useless_edges.insert(edge);
  });
  for (SSTaskEdge* edge : useless_edges) { DisConnect(edge); }
}

SSTaskGraph::SSTaskGraph(const Plan& plan,
			 const std::list<ActEvent>& act_events)
    : plan_(&plan) {
  InitGraph();
  InitUtilization(act_events);
}

void SSTaskGraph::MakeTaskId2AvgDurationHash(
    std::unordered_map<uint64_t, float>* task_id2avg_duration) const {
  for (const auto& pair : task_id2utilization_protos_) {
    (*task_id2avg_duration)[pair.first] =
        ConsumingTimePerPiece(pair.second) * 3;
  }
}

float SSTaskGraph::InitiationInterval() const {
  std::unordered_map<std::string, float> stream2consuming_per_piece;
  for (const auto& pair : stream2utilization_protos_) {
    stream2consuming_per_piece[pair.first] = ConsumingTimePerPiece(pair.second);
  }
  float ii = 0;
  for (const auto& pair : task_id2utilization_protos_) {
    float sum = 0;
    for (const std::string& stream : task_id2streams_.at(pair.first)) {
      sum += stream2consuming_per_piece.at(stream);
    }
    float avg_consumming_per_piece =
        sum / task_id2streams_.at(pair.first).size();
    float candidate_ii =
        avg_consumming_per_piece / task_id2streams_.at(pair.first).size();
    ii = std::max(ii, candidate_ii);
  }
  return ii;
}

void SSTaskGraph::InitUtilization(const std::list<ActEvent>& act_events) {
  std::list<UtilizationEventPackageProto> event_packages;
  GetUtilizationEventPackages(event_package_dir, fs, &event_packages);
  double last_event_end_at = 0;
  int64_t last_piece_id = 0;
  for (const auto& evt_pkg : event_packages) {
    TaskType task_type =
        GetSSTaskNode(evt_pkg.event(0).resource().task_stream().task_id())
            ->type();
    if (task_type == TaskType::kDataCompTask) {
      last_event_end_at =
          std::max(last_event_end_at, PackageLastEventEndAt(evt_pkg));
      last_piece_id = std::max(last_piece_id, PackageLastPieceId(evt_pkg));
    }
  }

  for (const auto& evt_pkg : event_packages) {
    if (PackageLastEventEndAt(evt_pkg) <= last_event_end_at
        && PackageLastPieceId(evt_pkg) == last_piece_id) {
      UtilizationMgr::Singleton()->GetUtilizationPackageFromEvent(
          evt_pkg, &utilization_package_);
    }
  }
  InitUtilizationIndexes();
}

void SSTaskGraph::InitUtilizationIndexes() {
  for (const auto& u : utilization_package_.utilization()) {
    uint64_t task_id = u.resource().task_stream().task_id();
    uint64_t stream_id = u.resource().task_stream().stream_id();
    std::string stream =
        GetSSTaskNode(task_id)->GlobalUniqueStreamName(stream_id);
    task_id2utilization_protos_[task_id].push_back(&u);
    stream2utilization_protos_[stream].push_back(&u);
    task_id2streams_[task_id].insert(stream);
  }
}

void SSTaskGraph::InitGraph() {
  for (const TaskProto& task_proto : plan().task()) {
    SSTaskNode* task = new SSTaskNode(task_proto);
    EnrollNode(task);
    CHECK(task_id2task_.emplace(task->task_id(), task).second);
  }
  ForEachEdgeFromPlan(plan(), [&](uint64_t producer_id, uint64_t consumer_id) {
    SSTaskNode* producer = task_id2task_[producer_id];
    SSTaskNode* consumer = task_id2task_[consumer_id];
    SSTaskEdge* edge = NewEdge();
    Connect(producer, edge, consumer);
  });
  UpdateSourceAndSink();
  UpdateAncestors();
  RemoveMeaninglessEdges();
}

}  // namespace oneflow
