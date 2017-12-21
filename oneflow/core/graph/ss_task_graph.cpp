#include "oneflow/core/graph/ss_task_graph.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

// namespace {
//
// oid ForEachEdgeFromPlan(
//    const Plan& plan,
//    const std::function<void(uint64_t, uint64_t)>& DoEachEdge) {
//  std::unordered_map<int64_t, const TaskProto*> id2task_proto;
//  for (const TaskProto& task_proto : plan.task()) {
//    id2task_proto[task_proto.task_id()] = &task_proto;
//  }
//  for (const TaskProto& task_proto : plan.task()) {
//    for (const auto& pair : task_proto.produced_regst_desc()) {
//      for (int64_t consumer_id : pair.second.consumer_task_id()) {
//        //  avoid circle
//        if (task_proto.task_type() == TaskType::kMdUpdt
//            && id2task_proto[consumer_id]->task_type() == TaskType::kForward)
//            {
//          continue;
//        }
//        DoEachEdge(task_proto.task_id(), consumer_id);
//      }
//    }
//  }
//
//
// ouble ConsumingTimePerPiece(const std::list<const ActEvent*>& act_events) {
//  std::unordered_set<int64_t> act_ids;
//  double total_consumming = 0;
//  for (const ActEvent* act_event : act_events) {
//    total_consumming += act_event->stop_time() - act_event->start_time();
//    act_ids.insert(act_event->act_id());
//  }
//  CHECK(act_ids.size());
//  return total_consumming / act_ids.size();
//
//
//}  // namespace

// std::string SSTaskNode::GetDeviceName() const {
//   uint64_t thrd_loc_id =
//       IDMgr::Singleton()->ThrdId4ActorId(task_proto().task_id());
//   std::string thrd_loc_name = "";
//   if (thrd_loc_id == IDMgr::Singleton()->CommNetThrdId()) {
//     thrd_loc_name = "comm_net";
//   } else if (thrd_loc_id >= JobDesc::Singleton()->PersistenceWorkerNum()) {
//     thrd_loc_name = "boxing";
//   } else if (thrd_loc_id
//              >= JobDesc::Singleton()->resource().device_num_per_machine()) {
//     thrd_loc_name = "persistence";
//   } else {
//     thrd_loc_name = std::to_string(thrd_loc_id);
//     if (task_proto().task_type() == TaskType::kCopyHd) {
//       thrd_loc_name = thrd_loc_name + ":copy_hd";
//     }
//   }
//   return IDMgr::Singleton()->MachineName4MachineId(
//              IDMgr::Singleton()->MachineId4ActorId(task_proto().task_id()))
//          + ":" + thrd_loc_name;
// }
//
// std::string SSTaskNode::VisualStr() const {
//   std::string name = std::to_string(task_proto_->task_id()) + "\\n";
//   if (task_proto_->task_type() != TaskType::kForward
//       || task_proto_->task_type() != TaskType::kBackward) {
//     name = name + TaskType_Name(task_proto_->task_type());
//   } else {
//     for (int i = 0; i < task_proto_->exec_sequence().exec_node_size(); ++i) {
//       name = name
//              + task_proto_->exec_sequence()
//                    .exec_node(i)
//                    .kernel_conf()
//                    .op_conf()
//                    .name()
//              + "\\n";
//     }
//   }
//   return name;
// }
//
// void SSTaskGraph::ForEachRegstDesc(
//     const std::function<void(const RegstDescProto&)>& DoEach) const {
//   for (const TaskProto& task_proto : plan().task()) {
//     for (const auto& pair : task_proto.produced_regst_desc()) {
//       DoEach(pair.second);
//     }
//   }
// }
//
// bool SSTaskGraph::IsAncestor(const SSTaskNode* asc,
//                              const SSTaskNode* node) const {
//   return task2ancestors_.at(node).find(asc) !=
//   task2ancestors_.at(node).end();
// }

void SSTaskGraph::MakeRegstDescId2AvgLifeTimeHash(
    std::unordered_map<uint64_t, double>* regst_desc_id2life_time,
    const std::function<double(uint64_t)>& AvgDuration4TaskId) const {
  // TODO

  // auto ForEachNext = [&](const SSTaskNode* task,
  //                        const std::function<void(const SSTaskNode*)>&
  //                        DoEach) {
  //   for (SSTaskEdge* edge : task->out_edges()) { DoEach(edge->dst_node()); }
  // };
  // auto ForEachPrev = [&](const SSTaskNode* task,
  //                        const std::function<void(const SSTaskNode*)>&
  //                        DoEach) {
  //   for (SSTaskEdge* edge : task->in_edges()) { DoEach(edge->src_node()); }
  // };
  // auto IsAscendant = std::bind(&SSTaskGraph::IsAncestor, this,
  //                              std::placeholders::_1, std::placeholders::_2);
  // LongestPathVisitor<const SSTaskNode*> lpath_visitor(ForEachNext,
  // ForEachPrev,
  //                                                     IsAscendant);
  // auto AvgDuration4Task = [&](const SSTaskNode* task) -> double {
  //   return AvgDuration4TaskId(task->task_id());
  // };
  // ForEachRegstDesc([&](const RegstDescProto& regst_desc) {
  //   double life_time = 0;
  //   auto producer = GetSSTaskNode(regst_desc.producer_task_id());
  //   CHECK(producer);
  //   for (int64_t consumer_id : regst_desc.consumer_task_id()) {
  //     auto consumer = GetSSTaskNode(consumer_id);
  //     life_time = std::max(life_time, AvgLifeTime(lpath_visitor, producer,
  //                                                 consumer,
  //                                                 AvgDuration4Task));
  //   }
  //   regst_desc_id2life_time->emplace(regst_desc.regst_desc_id(), life_time);
  // });
}

// double SSTaskGraph::AvgLifeTime(
//     const LongestPathVisitor<const SSTaskNode*>& lpath_visitor,
//     const SSTaskNode* start_task, const SSTaskNode* end_task,
//     const std::function<double(const SSTaskNode* task)>& AvgDuration4Task)
//     const {
//   double life_time = 0;
//   lpath_visitor(start_task, end_task, AvgDuration4Task,
//                 [&](const std::list<const SSTaskNode*>& path) {
//                   if (path.back() == end_task) {
//                     double d = 0;
//                     for (auto task : path) { d += AvgDuration4Task(task); }
//                     life_time = std::max(life_time, d);
//                   }
//                 });
//   return life_time;
// }
//
// void SSTaskGraph::UpdateAncestors() {
//   TopoForEachNode([&](SSTaskNode* task) {
//     for (const SSTaskEdge* edge : task->in_edges()) {
//       const SSTaskNode* in_task = edge->src_node();
//       for (const SSTaskNode* ancestor : task2ancestors_[in_task]) {
//         task2ancestors_[task].insert(ancestor);
//       }
//       task2ancestors_[task].insert(in_task);
//     }
//   });
// }
//
// bool SSTaskGraph::ReachableWithoutEdge(const SSTaskEdge* edge) const {
//   for (const SSTaskEdge* in_edge_of_dst : edge->dst_node()->in_edges()) {
//     const SSTaskNode* in_node_of_dst = in_edge_of_dst->src_node();
//     if (task2ancestors_.at(in_node_of_dst).find(edge->src_node())
//         != task2ancestors_.at(in_node_of_dst).end()) {
//       return true;
//     }
//   }
//   return false;
// }
//
// void SSTaskGraph::RemoveMeaninglessEdges() {
//   std::unordered_set<SSTaskEdge*> useless_edges;
//   ForEachEdge([&](SSTaskEdge* edge) {
//     if (ReachableWithoutEdge(edge)) useless_edges.insert(edge);
//   });
//   for (SSTaskEdge* edge : useless_edges) { DisConnect(edge); }
// }

SSTaskGraph::SSTaskGraph(const Plan& plan,
                         std::unique_ptr<std::list<ActEvent>>&& act_events)
    : plan_(&plan), act_events_(std::move(act_events)) {
  // InitGraph();
  // InitActEvents();
}

void SSTaskGraph::MakeTaskId2AvgDurationHash(
    std::unordered_map<uint64_t, double>* task_id2avg_duration) const {
  // TODO

  // for (const auto& pair : task_id2act_events_) {
  //   (*task_id2avg_duration)[pair.first] = ConsumingTimePerPiece(pair.second);
  // }
}

double SSTaskGraph::InitiationInterval() const {
  // TODO
  return 0;

  // double ii = 0;
  // for (const auto& pair : stream_id2act_events_) {
  //   ii = std::max(ii, ConsumingTimePerPiece(pair.second));
  // }
  // CHECK(ii);
  // return ii;
}

// void SSTaskGraph::InitActEvents() {
//   for (const ActEvent& act_event : *act_events_) {
//     int64_t task_id = act_event.actor_id();
//     int64_t stream_id = act_event.work_stream_id();
//     task_id2act_events_[task_id].push_back(&act_event);
//     stream_id2act_events_[stream_id].push_back(&act_event);
//   }
// }
//
// void SSTaskGraph::InitGraph() {
//   for (const TaskProto& task_proto : plan().task()) {
//     SSTaskNode* task = new SSTaskNode(task_proto);
//     AddAllocatedNode(task);
//     CHECK(task_id2task_.emplace(task->task_id(), task).second);
//   }
//   ForEachEdgeFromPlan(plan(), [&](uint64_t producer_id, uint64_t consumer_id)
//   {
//     SSTaskNode* producer = task_id2task_[producer_id];
//     SSTaskNode* consumer = task_id2task_[consumer_id];
//     SSTaskEdge* edge = NewEdge();
//     Connect(producer, edge, consumer);
//   });
//   // UpdateSourceAndSink();
//   UpdateAncestors();
//   RemoveMeaninglessEdges();
// }

}  // namespace oneflow
