/**
 * Copyright 2017 Xinqi Li
 */
#include "oneflow/core/schedule/simulator.h"
#include "oneflow/core/schedule/node.h"

namespace oneflow {
namespace schedule {

float SessionLogger::GetDurationByTimeGapToLoss(TaskInstance* from,
                                                TaskInstance* to) {
  float duration = 0.0;
  auto to_loss_gaps = mut_end_time_gap_to_loss()[to];
  for (const auto& from_loss_gap : mut_start_time_gap_to_loss()[from]) {
    auto to_loss_gap_itt = to_loss_gaps.find(from_loss_gap.first);
    if (to_loss_gap_itt == to_loss_gaps.end()) continue;
    float d = to_loss_gap_itt->second - from_loss_gap.second;
    duration = std::max(duration, d);
  }
  return duration;
}

void SessionLogger::UpdateDuration(SimulatorSession* session, Mode* strategy) {
  session->graph()->ForeachRegstDesc([&](RegstDesc* regst_desc) {
    Node* owner = nullptr;
    session->graph()->produced_regst_desc_mgr().Input(regst_desc, &owner);
    float duration = 0;
    uint32_t start = session->nr_base_batch();
    uint32_t end = start + session->nr_base_batch();
    //    uint32_t start = 0;
    //    uint32_t end = start + 1;
    session->graph()->subscribed_regst_desc_mgr().Input(
        regst_desc, [&](Node* node) {
          float sum = 0;
          for (uint32_t i = start; i < end; i++) {
            auto batch = session->batch_node_mgr().Find(i);
            TaskInstance* owner_instance =
                session->batch_arc_mgr().Find(batch, owner);
            TaskInstance* node_instance =
                session->batch_arc_mgr().Find(batch, node);
            sum += GetDurationByTimeGapToLoss(owner_instance, node_instance);
          }
          float avg = sum / std::max(1u, (end - start));
          duration = std::max(duration, avg);
        });
    mut_regst_desc2duration()[regst_desc] = std::round(duration);
  });
}

void LazyStrategy::TimeLinePushBack(TaskInstance* instance,
                                    DeviceNode* device) {
  auto last = dev2current_instance_[device];
  if (last) { mut_timenet_arc_mgr().CreateIfNotFound(last, instance); }
  dev2current_instance_[device] = instance;
}

void SessionLogger::UpdateInterval(SimulatorSession* session, Mode* strategy) {
  session->graph()->ForeachNode([&](Node* node) {
    uint32_t sum = 0;
    uint32_t last = 0;
    uint32_t start = session->nr_base_batch();
    uint32_t end = start + session->nr_base_batch();
    for (uint32_t i = start; i < end; i++) {
      auto batch = session->batch_node_mgr().Find(i);
      auto instance = session->batch_arc_mgr().Find(batch, node);
      auto start = strategy->GetTime(mut_instance2ended_at()[instance].first);
      if (last) { sum += start - last; }
      last = start;
    }
    mut_node2interval()[node] = 1.0 * sum / (end - 1 - start);
  });
  session->graph()->ForeachNode([&](Node* node) {
    mut_max_interval() = std::max(max_interval(), mut_node2interval()[node]);
  });
}

void SimulatorSession::ClearTmpData() {
  tokens_.clear();
  logger()->Clear();
}

void SessionLogger::Clear() {
  mut_instance2ended_at().clear();
  mut_device2ended_at().clear();
  mut_node2interval().clear();
  mut_regst_desc2duration().clear();
}

void SessionLogger::MergeTimeGapToLossInPlace(SessionLogger* logger) {
  typedef decltype(start_time_gap_to_loss_) TimeGap;
  auto merge = [&](TimeGap* a, TimeGap* b) {
    for (auto& a_loss2duration : *a) {
      auto b_loss_duration_itt = b->find(a_loss2duration.first);
      if (b_loss_duration_itt == b->end()) continue;
      for (auto& a_duration : a_loss2duration.second) {
        auto b_duration_itt =
            b_loss_duration_itt->second.find(a_duration.first);
        if (b_duration_itt == b_loss_duration_itt->second.end()) continue;
        if (std::abs(b_duration_itt->second) < std::abs(a_duration.second)) {
          a_duration.second = b_duration_itt->second;
        }
      }
    }
  };
  merge(&mut_start_time_gap_to_loss(), &logger->mut_start_time_gap_to_loss());
  merge(&mut_end_time_gap_to_loss(), &logger->mut_end_time_gap_to_loss());
}

void SessionLogger::UpdateTimeGapToLoss(SimulatorSession* session,
                                        Mode* strategy) {
  std::list<Node*> loss_nodes;
  session->graph()->LossNodes(&loss_nodes);
  uint32_t start = 0;
  uint32_t end = start + session->nr_batch();
  for (uint32_t i = start; i < end; i++) {
    auto batch = session->batch_node_mgr().Find(i);
    for (Node* loss : loss_nodes) {
      auto loss_instance = session->batch_arc_mgr().Find(batch, loss);
      auto loss_start_time =
          strategy->GetStartTime(mut_instance2ended_at()[loss_instance]);
      auto loss_end_time =
          strategy->GetEndTime(mut_instance2ended_at()[loss_instance]);
      float loss_middle_time =
          ((float)loss_start_time + (float)loss_end_time) / 2;
      auto set_time_gap = [&](Node* node) {
        auto node_instance = session->batch_arc_mgr().Find(batch, node);
        float start_time =
            strategy->GetStartTime(mut_instance2ended_at()[node_instance]);
        float end_time =
            strategy->GetEndTime(mut_instance2ended_at()[node_instance]);
        mut_start_time_gap_to_loss()[node_instance][loss] =
            start_time - loss_middle_time;
        mut_end_time_gap_to_loss()[node_instance][loss] =
            end_time - loss_middle_time;
      };
      set_time_gap(loss);
      session->graph()->ForeachAscendent(loss, set_time_gap);
      session->graph()->ForeachDescendent(loss, set_time_gap);
    }
  }
}

void SimulatorSession::NewSinkTokens() {
  ClearTmpData();
  std::list<Node*> places;
  graph()->arc_mgr().InputArc(graph()->sink(), [&](TaskArc* arc) {
    places.push_back(dynamic_cast<Node*>(arc));
  });
  auto batchs = GetBatchNodes();
  batch_arc_mgr().Find(*batchs, places,
                       [&](TaskInstance* arc) { tokens_.insert(arc); });
  InitNodeBatchInstance(graph()->sink());
}

void SimulatorSession::InitNodeBatchInstance(Node* node) {
  Session::InitNodeBatchInstance(node);
  for (uint32_t i = 0; i < nr_batch(); i++) {
    auto batch = batch_node_mgr().Find(i);
    auto start_instance = mut_batch_arc_mgr().Find(batch, node);
    logger()->mut_instance2ended_at()[start_instance] = std::make_pair(0u, 0u);
  }
}

void SimulatorSession::NewSourceTokens() {
  ClearTmpData();
  std::list<Node*> places;
  graph()->arc_mgr().OutputArc(graph()->source(), [&](TaskArc* arc) {
    places.push_back(dynamic_cast<Node*>(arc));
  });
  auto batchs = GetBatchNodes();
  batch_arc_mgr().Find(*batchs, places,
                       [&](TaskInstance* arc) { tokens_.insert(arc); });
  InitNodeBatchInstance(graph()->source());
}

DeviceNode* SimulatorSession::GetInstanceDevice(TaskInstance* instance) {
  DeviceNode* ret = nullptr;
  graph()->device_arc_mgr().Output(instance->to(), &ret);
  return ret;
}

int PositiveStrategy::HoldingRegstDesc(
    Node* node, const std::function<void(RegstDesc*)>& cb) {
  return Sess()->graph()->produced_regst_desc_mgr().Output(node, cb);
}

int PositiveStrategy::RegstDescReleasingNode(
    RegstDesc* regst_desc, const std::function<void(Node*)>& cb) {
  return Sess()->graph()->subscribed_regst_desc_mgr().Input(regst_desc, cb);
}

int NegativeStrategy::HoldingRegstDesc(
    Node* node, const std::function<void(RegstDesc*)>& cb) {
  return Sess()->graph()->subscribed_regst_desc_mgr().Output(node, cb);
}

int NegativeStrategy::RegstDescReleasingNode(
    RegstDesc* regst_desc, const std::function<void(Node*)>& cb) {
  return Sess()->graph()->produced_regst_desc_mgr().Input(regst_desc, cb);
}

bool PositiveStrategy::CompareInstanceOrder(TaskInstance* instance_a,
                                            TaskInstance* instance_b) {
  if (instance_a->to() == instance_b->to()) {
    // same node
    return instance_a->from()->id() < instance_b->from()->id();
  }
  if (instance_a->from() == instance_b->from()) {
    // same batch
    return instance_a->to()->depth() > instance_b->to()->depth();
  }
  return instance_a->to()->depth() < instance_b->to()->depth();
}

bool NegativeStrategy::CompareInstanceOrder(TaskInstance* instance_a,
                                            TaskInstance* instance_b) {
  if (instance_a->to() == instance_b->to()) {
    // same node
    return instance_a->from()->id() > instance_b->from()->id();
  }
  if (instance_a->from() == instance_b->from()) {
    // same batch
    return instance_a->to()->depth() < instance_b->to()->depth();
  }
  return instance_a->to()->depth() > instance_b->to()->depth();
}

TaskInstance* DirectionStrategy::PickInstanceToRun(
    const std::list<TaskInstance*>& instances) {
  TaskInstance* ret = nullptr;
  if (instances.size()) {
    auto itt = instances.begin();
    ret = *itt;
    for (; itt != instances.end(); itt++) {
      if (CompareInstanceOrder(*itt, ret)) { ret = *itt; }
    }
  }
  return ret;
}

void ResourceStrategy::InitFuncs() {
  get_node_instance_ = std::bind(&DirectionStrategy::GetNextNodeInstance,
                                 direction_, std::placeholders::_1);
  is_instance_ready_ = std::bind(&ResourceStrategy::IsInstanceReady, this,
                                 std::placeholders::_1);
  get_instance_device_ = std::bind(&SimulatorSession::GetInstanceDevice, Sess(),
                                   std::placeholders::_1);
  get_ascendent_ended_at_ = std::bind(&ResourceStrategy::GetAscendentEndedAt,
                                      this, std::placeholders::_1);
  pick_instance_to_run_ = std::bind(&DirectionStrategy::PickInstanceToRun,
                                    direction_, std::placeholders::_1);
}

TaskInstance* NegativeStrategy::GetNextNodeInstance(TaskInstance* arc) {
  auto input_arc = sess_->graph()->arc_mgr().Find(arc->to()->id());
  return sess_->batch_arc_mgr().Find(arc->from(), input_arc->from());
}

TaskInstance* PositiveStrategy::GetNextNodeInstance(TaskInstance* arc) {
  auto input_arc = sess_->graph()->arc_mgr().Find(arc->to()->id());
  return sess_->batch_arc_mgr().Find(arc->from(), input_arc->to());
}

void PositiveStrategy::NewStartTokens() { sess_->NewSourceTokens(); }

bool ResourceStrategy::IsInstanceReady(TaskInstance* instance) {
  bool ready = true;
  direction_->PrevArc(instance->to(), [&](TaskArc* arc) {
    auto place = dynamic_cast<Node*>(arc);
    auto instance_input = Sess()->batch_arc_mgr().Find(instance->from(), place);
    if (Sess()->tokens_.find(instance_input) == Sess()->tokens_.end()) {
      ready = false;
    }
  });
  return ready;
}

void NegativeStrategy::NewStartTokens() { sess_->NewSinkTokens(); }

unsigned int PositiveStrategy::PrevArc(
    Node* node, const std::function<void(TaskArc*)>& cb) {
  return sess_->graph()->arc_mgr().InputArc(node, cb);
}

unsigned int PositiveStrategy::Prev(Node* node,
                                    const std::function<void(Node*)>& cb) {
  return sess_->graph()->arc_mgr().Input(node, cb);
}

unsigned int PositiveStrategy::NextArc(
    Node* node, const std::function<void(TaskArc*)>& cb) {
  return sess_->graph()->arc_mgr().OutputArc(node, cb);
}

unsigned int PositiveStrategy::Next(Node* node,
                                    const std::function<void(Node*)>& cb) {
  return sess_->graph()->arc_mgr().Output(node, cb);
}

unsigned int NegativeStrategy::PrevArc(
    Node* node, const std::function<void(TaskArc*)>& cb) {
  return sess_->graph()->arc_mgr().OutputArc(node, cb);
}

unsigned int NegativeStrategy::Prev(Node* node,
                                    const std::function<void(Node*)>& cb) {
  return sess_->graph()->arc_mgr().Output(node, cb);
}

unsigned int NegativeStrategy::NextArc(
    Node* node, const std::function<void(TaskArc*)>& cb) {
  return sess_->graph()->arc_mgr().InputArc(node, cb);
}

unsigned int NegativeStrategy::Next(Node* node,
                                    const std::function<void(Node*)>& cb) {
  return sess_->graph()->arc_mgr().Input(node, cb);
}

void LimitedStrategy::InitFuncIsInstanceReady() {
  is_instance_ready_ = [&](TaskInstance* instance) {
    return IsInstanceReady(instance) && IsAllRegstDescReady(instance);
  };
  get_ascendent_ended_at_ = [&](TaskInstance* instance) {
    return std::max(evaluation_->GetAscendentEndedAt(instance),
                    RegstDescEndedAt(instance));
  };
}

void LazyStrategy::WalkTimeNetReverse(
    const std::function<void(TaskInstance*)>& cb) {
  auto last_batch = direction_->EndBatch();
  auto last_node = direction_->EndNode();
  auto last_instance = Sess()->batch_arc_mgr().Find(last_batch, last_node);
  auto next = std::unordered_set<TaskInstance*>{last_instance};
  auto marked = std::unordered_set<TaskInstance*>{};
  while (next.size()) {
    auto queue = std::list<TaskInstance*>(next.begin(), next.end());
    for (const auto& instance : queue) {
      cb(instance);
      marked.insert(instance);
      next.erase(instance);
      timenet_arc_mgr().Input(instance, [&](TaskInstance* prev) {
        //        std::cout << "prev\t" << prev->name()
        //          << " -> " << node->name()
        //          << std::endl;
        bool all_marked = true;
        timenet_arc_mgr().Output(prev, [&](TaskInstance* to) {
          if (all_marked && marked.find(to) == marked.end()) {
            all_marked = false;
          }
        });
        if (all_marked && marked.find(prev) == marked.end()) {
          next.insert(prev);
        }
      });
    }
  }
}

void LazyStrategy::Retiming() {
  float max_interval = Sess()->logger()->max_interval();
  auto get_next_instance = [&](TaskInstance* instance) {
    TaskInstance* next = nullptr;
    if (instance->to() != Sess()->graph()->sink()) {
      auto batch = instance->from();
      auto next_batch_id = direction_->NextBatchId(batch->id());
      auto next_batch = Sess()->batch_node_mgr().Find(next_batch_id);
      next = Sess()->batch_arc_mgr().Find(next_batch, instance->to());
    }
    return next;
  };
  WalkTimeNetReverse([&](TaskInstance* instance) {
    auto lazy_end = INT_MAX;
    int count = timenet_arc_mgr().Output(instance, [&](TaskInstance* instance) {
      const auto& p = Sess()->logger()->mut_instance2ended_at()[instance];
      lazy_end = std::min(lazy_end, p.first);
    });
    auto& p = Sess()->logger()->mut_instance2ended_at()[instance];
    if (!count) {
      //      lazy_end = p.second + max_interval;
      lazy_end = p.second;
    }
    auto next_instance = get_next_instance(instance);
    if (next_instance) {
      auto next_instance_end =
          Sess()->logger()->mut_instance2ended_at()[next_instance].second;
      lazy_end = std::min((float)lazy_end, next_instance_end - max_interval);
    }
    lazy_end = std::max(lazy_end, p.second);
    auto lazy_start = lazy_end - (p.second - p.first);
    p.second = lazy_end;
    p.first = lazy_start;
    //    std::cout << instance->name()
    //      << "\t" << p.first << std::endl;
  });
}

void LazyStrategy::InitTimeNet() {
  Sess()->graph()->ForeachArc([&](TaskArc* arc) {
    uint32_t start = 0;
    uint32_t end = Sess()->nr_batch();
    for (uint32_t i = start; i < end; i++) {
      auto batch = Sess()->batch_node_mgr().Find(i);
      auto from_node = direction_->GetFrom(arc);
      auto to_node = direction_->GetTo(arc);
      auto from = Sess()->batch_arc_mgr().Find(batch, from_node);
      auto to = Sess()->batch_arc_mgr().Find(batch, to_node);
      mut_timenet_arc_mgr().CreateIfNotFound(from, to);
    }
  });
}

void LimitedStrategy::InitRegst(
    const std::function<uint64_t(uint32_t)>& get_regst_num) {
  Sess()->graph()->ForeachRegstDesc([&](RegstDesc* regst_desc) {
    auto count = get_regst_num(regst_desc->id());
    for (uint32_t i = 0; i < count; i++) {
      auto regst =
          mut_regst_node_mgr().Create(std::to_string(regst_desc->id()));
      mut_r2rd_arc_mgr().CreateIfNotFound(regst, regst_desc);
    }
  });
}

int32_t EvaluationStrategy::GetAscendentEndedAt(TaskInstance* instance) {
  int32_t ended_at = 0;
  direction_->Prev(instance->to(), [&](Node* node) {
    auto instance_input = Sess()->batch_arc_mgr().Find(instance->from(), node);
    auto itt = Sess()->logger()->instance2ended_at().find(instance_input);
    auto token_ended_at = INT_MAX;
    if (itt != Sess()->logger()->instance2ended_at().end()) {
      token_ended_at = itt->second.second;
    }
    ended_at = std::max(ended_at, token_ended_at);
  });
  auto dev = Sess()->GetInstanceDevice(instance);
  return std::max(ended_at, Sess()->logger()->mut_device2ended_at()[dev]);
}

int32_t ResourceStrategy::GetAscendentEndedAt(TaskInstance* instance) {
  return evaluation_->GetAscendentEndedAt(instance);
}

int32_t LimitedStrategy::RegstDescEndedAt(TaskInstance* instance) {
  int32_t ended_at = 0;
  direction_->HoldingRegstDesc(instance->to(), [&](RegstDesc* regst_desc) {
    auto regst = FindFreeRegst(regst_desc, instance->from());
    ended_at = std::max(ended_at, regst2ended_at_[regst]);
  });
  return ended_at;
}

void LimitedStrategy::BeforeRun(TaskInstance* instance) {
  direction_->HoldingRegstDesc(instance->to(), [&](RegstDesc* regst_desc) {
    auto regst = FindFreeRegst(regst_desc, instance->from());
    auto regst_desc_instance =
        Sess()->batch_arc_mgr().Find(instance->from(), regst_desc);
    if (!regst) {
      // BUG
      return;
    }
    regst_desc_instance2regst_[regst_desc_instance] = regst;
    direction_->RegstDescReleasingNode(regst_desc, [&](Node* node) {
      Node* subscriber_node =
          Sess()->batch_arc_mgr().Find(instance->from(), node);
      mut_regst_arc_mgr().CreateIfNotFound(subscriber_node, regst);
    });
  });
}

void LimitedStrategy::AfterRun(TaskInstance* instance) {
  std::list<Arc<Node, Regst>*> occupied_arcs;
  Node* instance_node = instance;
  regst_arc_mgr().OutputArc(instance_node, &occupied_arcs);
  for (auto arc : occupied_arcs) {
    regst2ended_at_[arc->to()] =
        Sess()->logger()->mut_instance2ended_at()[instance].second;
    mut_regst_arc_mgr().Delete(arc->id());
  }
}

bool LimitedStrategy::IsAllRegstDescReady(TaskInstance* instance) {
  bool all_ready = true;
  direction_->HoldingRegstDesc(instance->to(), [&](RegstDesc* regst_desc) {
    all_ready = (all_ready && IsRegstDescReady(regst_desc, instance->from()));
  });
  return all_ready;
}

bool LimitedStrategy::IsRegstFree(Regst* regst) {
  return regst_arc_mgr().Input(regst) == 0;
}

bool LimitedStrategy::IsRegstDescReady(RegstDesc* regst_desc, Batch* batch) {
  auto regst_desc_instance = Sess()->batch_arc_mgr().Find(batch, regst_desc);
  bool free = regst_desc_instance2regst_[regst_desc_instance];
  if (!free) {
    r2rd_arc_mgr().Input(
        regst_desc, [&](Regst* regst) { free = (free || IsRegstFree(regst)); });
  }
  return free;
}

Regst* LimitedStrategy::FindFreeRegst(RegstDesc* regst_desc, Batch* batch) {
  auto regst_desc_instance = Sess()->batch_arc_mgr().Find(batch, regst_desc);
  Regst* ret = regst_desc_instance2regst_[regst_desc_instance];
  if (!ret) {
    int32_t ended_at = INT_MAX;
    r2rd_arc_mgr().Input(regst_desc, [&](Regst* regst) {
      if (IsRegstFree(regst)) {
        if (regst2ended_at_[regst] < ended_at) {
          // first recycled register
          ended_at = regst2ended_at_[regst];
          ret = regst;
        }
      }
    });
  }
  return ret;
}

std::unique_ptr<std::unordered_map<DeviceNode*, TaskInstance*>>
ResourceStrategy::Pick(std::unordered_set<TaskInstance*>* tokens) {
  auto arc_id2tokens = XGroupBy<uint64_t>(
      *tokens, [](TaskInstance* instance) { return instance->to()->id(); });
  auto all_instances = XDistinct<TaskInstance*>(*tokens, get_node_instance_);
  auto ready_instances =
      XFilter<TaskInstance*>(*all_instances, is_instance_ready_);
  auto instances_groupby_ended_at =
      XGroupBy<int32_t>(*ready_instances, get_ascendent_ended_at_);
  auto first_finished = XAssocKMin(*instances_groupby_ended_at);
  auto instances_groupby_dev =
      XGroupBy<DeviceNode*>(first_finished->second, get_instance_device_);
  auto instances_picked =
      XAssocVMap<TaskInstance*>(*instances_groupby_dev, pick_instance_to_run_);
  return instances_picked;
}

void Mode::Run() {
  NewStartTokens();
  auto sess_logger = Sess()->logger();
  while (Sess()->tokens_.size()) {
    auto instances_picked = Pick(&Sess()->tokens_);
    for (const auto& p : *instances_picked) {
      auto dev = dynamic_cast<DeviceNode*>(p.first);
      auto batch = p.second->from();
      BeforeRun(p.second);
      int32_t ended_at = GetAscendentEndedAt(p.second);
      //      std::cout << p.second->name()
      //                << "\t" << direction_->GetTime(ended_at)
      //                << std::endl;
      sess_logger->mut_instance2ended_at()[p.second].first = ended_at;
      ended_at += (dev ? dev->time() : 0);
      sess_logger->mut_device2ended_at()[p.first] = ended_at;
      sess_logger->mut_instance2ended_at()[p.second].second = ended_at;
      TimeLinePushBack(p.second, dev);
      AfterRun(p.second);
      PrevArc(p.second->to(), [&](TaskArc* arc) {
        auto place = dynamic_cast<Node*>(arc);
        auto instance_input = Sess()->batch_arc_mgr().Find(batch, place);
        Sess()->tokens_.erase(instance_input);
      });
      NextArc(p.second->to(), [&](TaskArc* arc) {
        auto place = dynamic_cast<Node*>(arc);
        auto instance_output = Sess()->batch_arc_mgr().Find(batch, place);
        Sess()->tokens_.insert(instance_output);
      });
    }
    if (!instances_picked->size()) { break; }
  }
  sess_logger->UpdateInterval(Sess(), this);
  Retiming();
  sess_logger->UpdateTimeGapToLoss(Sess(), this);
  sess_logger->UpdateDuration(Sess(), this);
}

std::unique_ptr<Session> StaticSchedulerSimulatorPolicy::MakeSession(
    const SGraph& graph) {
  auto graph_ptr = const_cast<SGraph*>(&graph);
  return unique_ptr_new<SimulatorSession>(graph_ptr);
}

std::unique_ptr<ScheduleResult> StaticSchedulerSimulatorPolicy::Schedule(
    const Session& session) {
  auto session_ptr = const_cast<Session*>(&session);
  auto sess = dynamic_cast<SimulatorSession*>(session_ptr);

  UnlimitedMode<PositiveStrategy> m0(sess);
  m0.Run();

  return sess->GetLoggerThenReset();
}

void RetimingSimulatorPolicy::Retiming(const Session& session,
                                       ScheduleResult* result) {
  auto session_ptr = const_cast<Session*>(&session);
  auto sess = dynamic_cast<SimulatorSession*>(session_ptr);
  auto logger = dynamic_cast<SessionLogger*>(result);

  UnlimitedMode<NegativeStrategy> m1(sess);
  m1.Run();
  logger->MergeTimeGapToLossInPlace(&*sess->logger());
  logger->UpdateDuration(sess, &m1);
}

void AllocatorSimulatorPolicy::AllocateFromSchedule(const Session& session,
                                                    ScheduleResult* result) {
  auto session_ptr = const_cast<Session*>(&session);
  auto sess = dynamic_cast<SimulatorSession*>(session_ptr);
  auto logger = dynamic_cast<SessionLogger*>(result);

  sess->graph()->ForeachRegstDesc([&](RegstDesc* regst_desc) {
    Node* owner = nullptr;
    sess->graph()->produced_regst_desc_mgr().Input(regst_desc, &owner);
    auto duration = logger->mut_regst_desc2duration()[regst_desc];
    auto interval = logger->max_interval();
    auto count = (uint32_t)ceil(duration / std::max(interval, 1.0f));
    logger->mut_regst_desc2count()[regst_desc] = count;
    std::cout << "Allocation\t" << regst_desc->id() << "\t" << count << "\t"
              << duration << "," << interval << std::endl;
  });
}

}  // namespace schedule
}  // namespace oneflow
