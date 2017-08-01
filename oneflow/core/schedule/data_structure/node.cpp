/**
 * Copyright 2017 Xinqi Li
 */
#include "oneflow/core/schedule/data_structure/node.h"

namespace oneflow {
namespace schedule {

void GraphNode::InitSourceAndSink() {
  mut_source() = mut_fake_node_mgr().Create("source");
  mut_sink() = mut_fake_node_mgr().Create("sink");
}

int GraphNode::LossNodes(std::list<Node*>* l) const {
  return loss_arc_mgr().Output(this, l);
}

void GraphNode::UpdateSourceAndSink() {
  std::list<Arc*> arcs;
  arc_mgr().OutputArc(source(), &arcs);
  arc_mgr().InputArc(sink(), &arcs);
  for (auto arc : arcs) { mut_arc_mgr().Delete(arc->id()); }
  children_arc_mgr().Output(this, [&](Node* leaf) {
    if (!arc_mgr().Input(leaf)) {
      mut_arc_mgr().CreateIfNotFound(source(), leaf);
    }
    if (!arc_mgr().Output(leaf)) {
      mut_arc_mgr().CreateIfNotFound(leaf, sink());
    }
  });
}

void GraphNode::ForeachArc(const std::function<void(Arc*)>& cb) const {
  arc_mgr().OutputArc(source(), cb);
  children_arc_mgr().Output(
      this, [&](Node* child) { arc_mgr().OutputArc(child, cb); });
}

void GraphNode::ForeachNodeWithSourceAndSink(
    const std::function<void(Node*)>& cb) const {
  cb(source());
  ForeachNode(cb);
  cb(sink());
}
void GraphNode::ForeachNode(const std::function<void(Node*)>& cb) const {
  cb(source());
  children_arc_mgr().Output(this, cb);
  cb(sink());
}

uint32_t GraphNode::Depth() {
  auto depth = source()->depth();
  return depth ? depth - 1 : 0;
}

uint32_t GraphNode::DeviceCount() {
  std::unordered_set<Node*> devices;
  children_arc_mgr().Output(this, [&](Node* node) {
    Node* device = nullptr;
    device_arc_mgr().Output(node, &device);
    devices.insert(device);
  });
  return devices.size();
}

void GraphNode::WalkArcReverse(const std::function<void(Arc*)>& cb) {
  WalkReverse([&](Node* node) {
    arc_mgr().OutputArc(node, [&](Arc* arc) { cb(arc); });
  });
}

void GraphNode::WalkReverse(const std::function<void(Node*)>& cb) {
  auto next = std::unordered_set<Node*>{sink()};
  auto marked = std::unordered_set<Node*>{};
  while (next.size()) {
    auto queue = std::list<Node*>(next.begin(), next.end());
    for (const auto& node : queue) {
      cb(node);
      marked.insert(node);
      next.erase(node);
      arc_mgr().InputArc(node, [&](Arc* arc) {
        bool all_marked = true;
        arc_mgr().Output(arc->from(), [&](Node* from) {
          if (all_marked && marked.find(from) == marked.end()) {
            all_marked = false;
          }
        });
        if (all_marked && marked.find(arc->from()) == marked.end()) {
          next.insert(arc->from());
        }
      });
    }
  }
}

void GraphNode::WalkArc(const std::function<void(Arc*)>& cb) {
  Walk([&](Node* node) { arc_mgr().InputArc(node, cb); });
}

void GraphNode::Walk(const std::function<void(Node*)>& cb) {
  auto next = std::unordered_set<Node*>{source()};
  auto marked = std::unordered_set<Node*>{};
  while (next.size()) {
    auto queue = std::list<Node*>(next.begin(), next.end());
    for (const auto& node : queue) {
      cb(node);
      marked.insert(node);
      next.erase(node);
      arc_mgr().OutputArc(node, [&](Arc* arc) {
        bool all_marked = true;
        arc_mgr().Input(arc->to(), [&](Node* from) {
          if (all_marked && marked.find(from) == marked.end()) {
            all_marked = false;
          }
        });
        if (all_marked && marked.find(arc->to()) == marked.end()) {
          next.insert(arc->to());
        }
      });
    }
  }
}

void GraphNode::InitAscendentArc() {
  Walk([&](Node* node) {
    arc_mgr().Input(node, [&](Node* prev) {
      std::list<Node*> l;
      ascendent_arc_mgr().Output(prev, &l);
      for (Node* asc : l) {
        mut_ascendent_arc_mgr().CreateIfNotFound(node, asc);
      }
      mut_ascendent_arc_mgr().CreateIfNotFound(node, prev);
    });
  });
}

void GraphNode::ForeachAscendent(Node* node,
                                 const std::function<void(Node*)>& cb) const {
  ascendent_arc_mgr().Output(node, cb);
}

void GraphNode::ForeachDescendent(Node* node,
                                  const std::function<void(Node*)>& cb) const {
  ascendent_arc_mgr().Input(node, cb);
}

void GraphNode::InitDepth() {
  WalkReverse([&](Node* node) {
    int depth = -1;
    arc_mgr().Output(node,
                     [&](Node* to) { depth = std::max(depth, to->depth()); });
    node->mut_depth() = depth + 1;
  });
}

void Session::ForeachRegstDesc(const std::function<void(Node*)>& cb) const {
  root()->children_arc_mgr().Output(root(), [&](Node* node) {
    root()->produced_regst_desc_mgr().Output(node, cb);
  });
}

float SessionLogger::GetDurationByTimeGapToLoss(Arc* from, Arc* to) {
  float duration = 0.0;
  auto to_loss_gaps = end_time_gap_to_loss_[to];
  for (const auto& from_loss_gap : start_time_gap_to_loss_[from]) {
    auto to_loss_gap_itt = to_loss_gaps.find(from_loss_gap.first);
    if (to_loss_gap_itt == to_loss_gaps.end()) continue;
    float d = to_loss_gap_itt->second - from_loss_gap.second;
    duration = std::max(duration, d);
  }
  return duration;
}

void SessionLogger::UpdateDuration(Session* session, Mode* strategy) {
  session->ForeachRegstDesc([&](Node* regst_desc) {
    Node* owner = nullptr;
    session->root()->produced_regst_desc_mgr().Input(regst_desc, &owner);
    float duration = 0;
    uint32_t start = session->nr_base_batch_;
    uint32_t end = start + session->nr_base_batch_;
    //    uint32_t start = 0;
    //    uint32_t end = start + 1;
    session->root()->subscribed_regst_desc_mgr().Input(
        regst_desc, [&](Node* node) {
          float sum = 0;
          for (uint32_t i = start; i < end; i++) {
            auto batch = session->batch_node_mgr().Find(i);
            auto owner_instance = session->batch_arc_mgr().Find(batch, owner);
            auto node_instance = session->batch_arc_mgr().Find(batch, node);
            sum += GetDurationByTimeGapToLoss(owner_instance, node_instance);
          }
          float avg = sum / std::max(1u, (end - start));
          duration = std::max(duration, avg);
        });
    regst_desc2duration_[regst_desc] = std::round(duration);
  });
}

void LazyStrategy::TimeLinePushBack(InstanceArc* instance, DeviceNode* device) {
  auto last = dev2current_instance_[device];
  if (last) { mut_timenet_arc_mgr().CreateIfNotFound(last, instance); }
  dev2current_instance_[device] = instance;
}

void SessionLogger::UpdateInterval(Session* session, Mode* strategy) {
  session->root()->ForeachNode([&](Node* node) {
    uint32_t sum = 0;
    uint32_t last = 0;
    uint32_t start = session->nr_base_batch_;
    uint32_t end = start + session->nr_base_batch_;
    for (uint32_t i = start; i < end; i++) {
      auto batch = session->batch_node_mgr().Find(i);
      auto instance = session->batch_arc_mgr().Find(batch, node);
      auto start = strategy->GetTime(arc2ended_at_[instance].first);
      if (last) { sum += start - last; }
      last = start;
    }
    node2interval_[node] = 1.0 * sum / (end - 1 - start);
  });
  session->root()->ForeachNode([&](Node* node) {
    max_interval_ = std::max(max_interval_, node2interval_[node]);
  });
}

std::unique_ptr<Session::PipeCount> Session::RegstDescCount(bool bottleneck) {
  auto regst_desc2pipe_count = unique_ptr_new<Session::PipeCount>();
  std::cout << "interval=" << logger()->max_interval_ << std::endl;
  ForeachRegstDesc([&](Node* regst_desc) {
    Node* owner = nullptr;
    root()->produced_regst_desc_mgr().Input(regst_desc, &owner);
    auto& spec = (*regst_desc2pipe_count)[regst_desc->id()];
    spec.duration = logger()->regst_desc2duration_[regst_desc];
    spec.freq = logger()->max_interval_;
    spec.count = (uint32_t)ceil(spec.duration / std::max(spec.freq, 1.0f));
    //    spec.count = std::min(nr_base_batch_, spec.count);
    std::cout << "regst_desc2pipe_count\t" << regst_desc->id() << "\t"
              << spec.count << "\t" << spec.duration << "," << spec.freq
              << std::endl;
  });
  return regst_desc2pipe_count;
}

void Session::ClearTmpData() {
  tokens_.clear();
  logger()->Clear();
}

void SessionLogger::Clear() {
  arc2ended_at_.clear();
  device2ended_at_.clear();
  node2interval_.clear();
  regst_desc2duration_.clear();
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
  merge(&start_time_gap_to_loss_, &logger->start_time_gap_to_loss_);
  merge(&end_time_gap_to_loss_, &logger->end_time_gap_to_loss_);
}

void SessionLogger::UpdateTimeGapToLoss(Session* session, Mode* strategy) {
  std::list<Node*> loss_nodes;
  session->root()->LossNodes(&loss_nodes);
  uint32_t start = 0;
  uint32_t end = start + session->nr_batch_;
  for (uint32_t i = start; i < end; i++) {
    auto batch = session->batch_node_mgr().Find(i);
    for (Node* loss : loss_nodes) {
      auto loss_instance = session->batch_arc_mgr().Find(batch, loss);
      auto loss_start_time =
          strategy->GetStartTime(arc2ended_at_[loss_instance]);
      auto loss_end_time = strategy->GetEndTime(arc2ended_at_[loss_instance]);
      float loss_middle_time =
          ((float)loss_start_time + (float)loss_end_time) / 2;
      auto set_time_gap = [&](Node* node) {
        auto node_instance = session->batch_arc_mgr().Find(batch, node);
        float start_time = strategy->GetStartTime(arc2ended_at_[node_instance]);
        float end_time = strategy->GetEndTime(arc2ended_at_[node_instance]);
        start_time_gap_to_loss_[node_instance][loss] =
            start_time - loss_middle_time;
        end_time_gap_to_loss_[node_instance][loss] =
            end_time - loss_middle_time;
      };
      set_time_gap(loss);
      session->root()->ForeachAscendent(loss, set_time_gap);
      session->root()->ForeachDescendent(loss, set_time_gap);
    }
  }
}

std::unique_ptr<std::list<Node*>> Session::GetBatchNodes() {
  auto batchs = unique_ptr_new<std::list<Node*>>();
  for (uint32_t i = 0; i < nr_batch_; i++) {
    batchs->push_back(batch_node_mgr().Find(i));
  }
  return batchs;
}

void Session::NewSinkTokens() {
  ClearTmpData();
  std::list<Node*> places;
  root()->arc_mgr().InputArc(root()->sink(), [&](Arc* arc) {
    places.push_back(dynamic_cast<Node*>(arc));
  });
  auto batchs = GetBatchNodes();
  batch_arc_mgr().Find(*batchs, places, [&](Arc* arc) { tokens_.insert(arc); });
  InitNodeBatchInstance(root()->sink());
}

void Session::InitNodeBatchInstance(Node* node) {
  for (uint32_t i = 0; i < nr_batch_; i++) {
    auto batch = batch_node_mgr().Find(i);
    auto start_instance = mut_batch_arc_mgr().CreateIfNotFound(batch, node);
    logger()->arc2ended_at_[start_instance] = std::make_pair(0u, 0u);
  }
}

void Session::NewSourceTokens() {
  ClearTmpData();
  std::list<Node*> places;
  root()->arc_mgr().OutputArc(root()->source(), [&](Arc* arc) {
    places.push_back(dynamic_cast<Node*>(arc));
  });
  auto batchs = GetBatchNodes();
  batch_arc_mgr().Find(*batchs, places, [&](Arc* arc) { tokens_.insert(arc); });
  InitNodeBatchInstance(root()->source());
}

void Session::NewBatchs() {
  std::list<Node*> batch_nodes;
  for (int i = 0; i < nr_batch_; i++) {
    auto batch = mut_batch_node_mgr().CreateWithId(i, std::to_string(i));
    batch_nodes.push_back(batch);
  }
  root()->ForeachNodeWithSourceAndSink([&](Node* node) {
    for (auto batch : batch_nodes) {
      auto instance = mut_batch_arc_mgr().CreateIfNotFound(batch, node);
      mut_batch_instance_node_mgr().CreateWithId(
          instance->id(), std::to_string(instance->id()));
    }
  });
  root()->ForeachArc([&](Arc* arc) {
    auto place = dynamic_cast<Node*>(arc);
    for (auto batch : batch_nodes) {
      mut_batch_arc_mgr().CreateIfNotFound(batch, place);
    }
  });
  ForeachRegstDesc([&](Node* regst_desc) {
    for (auto batch : batch_nodes) {
      mut_batch_arc_mgr().CreateIfNotFound(batch, regst_desc);
    }
  });
}

Node* Session::GetInstanceDevice(Arc* instance) {
  Node* ret = nullptr;
  root()->device_arc_mgr().Output(instance->to(), &ret);
  return ret;
}

int PositiveStrategy::HoldingRegstDesc(Node* node,
                                       const std::function<void(Node*)>& cb) {
  return Sess()->root()->produced_regst_desc_mgr().Output(node, cb);
}

int PositiveStrategy::RegstDescReleasingNode(
    Node* regst_desc, const std::function<void(Node*)>& cb) {
  return Sess()->root()->subscribed_regst_desc_mgr().Input(regst_desc, cb);
}

int NegativeStrategy::HoldingRegstDesc(Node* node,
                                       const std::function<void(Node*)>& cb) {
  return Sess()->root()->subscribed_regst_desc_mgr().Output(node, cb);
}

int NegativeStrategy::RegstDescReleasingNode(
    Node* regst_desc, const std::function<void(Node*)>& cb) {
  return Sess()->root()->produced_regst_desc_mgr().Input(regst_desc, cb);
}

bool PositiveStrategy::CompareInstanceOrder(Arc* instance_a, Arc* instance_b) {
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

bool NegativeStrategy::CompareInstanceOrder(Arc* instance_a, Arc* instance_b) {
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

Arc* DirectionStrategy::PickInstanceToRun(const std::list<Arc*>& instances) {
  Arc* ret = nullptr;
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
  get_instance_device_ =
      std::bind(&Session::GetInstanceDevice, Sess(), std::placeholders::_1);
  get_ascendent_ended_at_ = std::bind(&ResourceStrategy::GetAscendentEndedAt,
                                      this, std::placeholders::_1);
  pick_instance_to_run_ = std::bind(&DirectionStrategy::PickInstanceToRun,
                                    direction_, std::placeholders::_1);
}

Arc* NegativeStrategy::GetNextNodeInstance(Arc* arc) {
  auto input_arc = sess_->root()->arc_mgr().Find(arc->to()->id());
  return sess_->batch_arc_mgr().Find(arc->from(), input_arc->from());
}

Arc* PositiveStrategy::GetNextNodeInstance(Arc* arc) {
  auto input_arc = sess_->root()->arc_mgr().Find(arc->to()->id());
  return sess_->batch_arc_mgr().Find(arc->from(), input_arc->to());
}

void PositiveStrategy::NewStartTokens() { sess_->NewSourceTokens(); }

bool ResourceStrategy::IsInstanceReady(Arc* instance) {
  bool ready = true;
  direction_->PrevArc(instance->to(), [&](Arc* arc) {
    auto place = dynamic_cast<Node*>(arc);
    auto instance_input = Sess()->batch_arc_mgr().Find(instance->from(), place);
    if (Sess()->tokens_.find(instance_input) == Sess()->tokens_.end()) {
      ready = false;
    }
  });
  return ready;
}

void NegativeStrategy::NewStartTokens() { sess_->NewSinkTokens(); }

unsigned int PositiveStrategy::PrevArc(Node* node,
                                       const std::function<void(Arc*)>& cb) {
  return sess_->root()->arc_mgr().InputArc(node, cb);
}

unsigned int PositiveStrategy::Prev(Node* node,
                                    const std::function<void(Node*)>& cb) {
  return sess_->root()->arc_mgr().Input(node, cb);
}

unsigned int PositiveStrategy::NextArc(Node* node,
                                       const std::function<void(Arc*)>& cb) {
  return sess_->root()->arc_mgr().OutputArc(node, cb);
}

unsigned int PositiveStrategy::Next(Node* node,
                                    const std::function<void(Node*)>& cb) {
  return sess_->root()->arc_mgr().Output(node, cb);
}

unsigned int NegativeStrategy::PrevArc(Node* node,
                                       const std::function<void(Arc*)>& cb) {
  return sess_->root()->arc_mgr().OutputArc(node, cb);
}

unsigned int NegativeStrategy::Prev(Node* node,
                                    const std::function<void(Node*)>& cb) {
  return sess_->root()->arc_mgr().Output(node, cb);
}

unsigned int NegativeStrategy::NextArc(Node* node,
                                       const std::function<void(Arc*)>& cb) {
  return sess_->root()->arc_mgr().InputArc(node, cb);
}

unsigned int NegativeStrategy::Next(Node* node,
                                    const std::function<void(Node*)>& cb) {
  return sess_->root()->arc_mgr().Input(node, cb);
}

void LimitedStrategy::InitFuncIsInstanceReady() {
  is_instance_ready_ = [&](Arc* instance) {
    return IsInstanceReady(instance) && IsAllRegstDescReady(instance);
  };
  get_ascendent_ended_at_ = [&](Arc* instance) {
    return std::max(evaluation_->GetAscendentEndedAt(instance),
                    RegstDescEndedAt(instance));
  };
}

void LazyStrategy::WalkTimeNetReverse(
    const std::function<void(InstanceArc*)>& cb) {
  auto last_batch = direction_->EndBatch();
  auto last_node = direction_->EndNode();
  auto last_instance = Sess()->batch_arc_mgr().Find(last_batch, last_node);
  auto init = dynamic_cast<Node*>(last_instance);
  auto next = std::unordered_set<Node*>{init};
  auto marked = std::unordered_set<Node*>{};
  while (next.size()) {
    auto queue = std::list<Node*>(next.begin(), next.end());
    for (const auto& node : queue) {
      auto instance = dynamic_cast<InstanceArc*>(node);
      cb(instance);
      marked.insert(node);
      next.erase(node);
      timenet_arc_mgr().Input(node, [&](Node* prev) {
        //        std::cout << "prev\t" << prev->name()
        //          << " -> " << node->name()
        //          << std::endl;
        bool all_marked = true;
        timenet_arc_mgr().Output(prev, [&](Node* to) {
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
  float max_interval = Sess()->logger()->max_interval_;
  auto get_next_instance = [&](InstanceArc* instance) {
    InstanceArc* next = nullptr;
    if (instance->to() != Sess()->root()->sink()) {
      auto batch = instance->from();
      auto next_batch_id = direction_->NextBatchId(batch->id());
      auto next_batch = Sess()->batch_node_mgr().Find(next_batch_id);
      next = Sess()->batch_arc_mgr().Find(next_batch, instance->to());
    }
    return next;
  };
  WalkTimeNetReverse([&](InstanceArc* instance) {
    auto lazy_end = INT_MAX;
    auto node = dynamic_cast<Node*>(instance);
    int count = timenet_arc_mgr().Output(node, [&](Node* node) {
      auto instance = dynamic_cast<InstanceArc*>(node);
      const auto& p = Sess()->logger()->arc2ended_at_[instance];
      lazy_end = std::min(lazy_end, p.first);
    });
    auto& p = Sess()->logger()->arc2ended_at_[instance];
    if (!count) {
      //      lazy_end = p.second + max_interval;
      lazy_end = p.second;
    }
    auto next_instance = get_next_instance(instance);
    if (next_instance) {
      auto next_instance_end =
          Sess()->logger()->arc2ended_at_[next_instance].second;
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
  Sess()->root()->ForeachArc([&](Arc* arc) {
    uint32_t start = 0;
    uint32_t end = Sess()->nr_batch_;
    for (uint32_t i = start; i < end; i++) {
      auto batch = Sess()->batch_node_mgr().Find(i);
      auto from_node = direction_->GetFrom(arc);
      auto to_node = direction_->GetTo(arc);
      auto from_arc = Sess()->batch_arc_mgr().Find(batch, from_node);
      auto to_arc = Sess()->batch_arc_mgr().Find(batch, to_node);
      auto from = dynamic_cast<Node*>(from_arc);
      auto to = dynamic_cast<Node*>(to_arc);
      mut_timenet_arc_mgr().CreateIfNotFound(from, to);
    }
  });
}

void LimitedStrategy::InitRegst(const Session::PipeCount& pipe_count) {
  Sess()->ForeachRegstDesc([&](Node* regst_desc) {
    auto pipe_count_itt = pipe_count.find(regst_desc->id());
    if (pipe_count_itt != pipe_count.end()) {
      const auto& spec = pipe_count_itt->second;
      for (uint32_t i = 0; i < spec.count; i++) {
        auto regst =
            mut_regst_node_mgr().Create(std::to_string(regst_desc->id()));
        mut_r2rd_arc_mgr().CreateIfNotFound(regst, regst_desc);
      }
    }
  });
}

int32_t EvaluationStrategy::GetAscendentEndedAt(Arc* instance) {
  int32_t ended_at = 0;
  direction_->Prev(instance->to(), [&](Node* node) {
    auto instance_input = Sess()->batch_arc_mgr().Find(instance->from(), node);
    auto itt = Sess()->logger()->arc2ended_at_.find(instance_input);
    auto token_ended_at = INT_MAX;
    if (itt != Sess()->logger()->arc2ended_at_.end()) {
      token_ended_at = itt->second.second;
    }
    ended_at = std::max(ended_at, token_ended_at);
  });
  auto dev = Sess()->GetInstanceDevice(instance);
  return std::max(ended_at, Sess()->logger()->device2ended_at_[dev]);
}

int32_t ResourceStrategy::GetAscendentEndedAt(Arc* instance) {
  return evaluation_->GetAscendentEndedAt(instance);
}

int32_t LimitedStrategy::RegstDescEndedAt(Arc* instance) {
  int32_t ended_at = 0;
  direction_->HoldingRegstDesc(instance->to(), [&](Node* regst_desc) {
    auto regst = FindFreeRegst(regst_desc, instance->from());
    ended_at = std::max(ended_at, regst2ended_at_[regst]);
  });
  return ended_at;
}

void LimitedStrategy::BeforeRun(Arc* instance) {
  direction_->HoldingRegstDesc(instance->to(), [&](Node* regst_desc) {
    auto regst = FindFreeRegst(regst_desc, instance->from());
    auto regst_desc_instance =
        Sess()->batch_arc_mgr().Find(instance->from(), regst_desc);
    if (!regst) {
      // BUG
      return;
    }
    regst_desc_instance2regst_[regst_desc_instance] = regst;
    direction_->RegstDescReleasingNode(regst_desc, [&](Node* node) {
      auto subscriber_arc =
          Sess()->batch_arc_mgr().Find(instance->from(), node);
      auto subscriber_node =
          Sess()->batch_instance_node_mgr().Find(subscriber_arc->id());
      mut_regst_arc_mgr().CreateIfNotFound(subscriber_node, regst);
    });
  });
}

void LimitedStrategy::AfterRun(Arc* instance) {
  std::list<Arc*> occupied_arcs;
  auto instance_node = Sess()->batch_instance_node_mgr().Find(instance->id());
  regst_arc_mgr().OutputArc(instance_node, &occupied_arcs);
  for (auto arc : occupied_arcs) {
    regst2ended_at_[arc->to()] =
        Sess()->logger()->arc2ended_at_[instance].second;
    mut_regst_arc_mgr().Delete(arc->id());
  }
}

bool LimitedStrategy::IsAllRegstDescReady(Arc* instance) {
  bool all_ready = true;
  direction_->HoldingRegstDesc(instance->to(), [&](Node* regst_desc) {
    all_ready = (all_ready && IsRegstDescReady(regst_desc, instance->from()));
  });
  return all_ready;
}

bool LimitedStrategy::IsRegstFree(Regst* regst) {
  return regst_arc_mgr().Input(regst) == 0;
}

bool LimitedStrategy::IsRegstDescReady(Node* regst_desc, Node* batch) {
  auto regst_desc_instance = Sess()->batch_arc_mgr().Find(batch, regst_desc);
  bool free = regst_desc_instance2regst_[regst_desc_instance];
  if (!free) {
    r2rd_arc_mgr().Input(
        regst_desc, [&](Regst* regst) { free = (free || IsRegstFree(regst)); });
  }
  return free;
}

Regst* LimitedStrategy::FindFreeRegst(Node* regst_desc, Node* batch) {
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

std::unique_ptr<std::unordered_map<Node*, Arc*>> ResourceStrategy::Pick(
    std::unordered_set<Arc*>* tokens) {
  auto arc_id2tokens = XGroupBy<uint64_t>(
      *tokens, [](Arc* instance) { return instance->to()->id(); });
  auto all_instances = XDistinct<Arc*>(*tokens, get_node_instance_);
  auto ready_instances = XFilter<Arc*>(*all_instances, is_instance_ready_);
  auto instances_groupby_ended_at =
      XGroupBy<int32_t>(*ready_instances, get_ascendent_ended_at_);
  auto first_finished = XAssocKMin(*instances_groupby_ended_at);
  auto instances_groupby_dev =
      XGroupBy<Node*>(first_finished->second, get_instance_device_);
  auto instances_picked =
      XAssocVMap<Arc*>(*instances_groupby_dev, pick_instance_to_run_);
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
      sess_logger->arc2ended_at_[p.second].first = ended_at;
      ended_at += (dev ? dev->time() : 0);
      sess_logger->device2ended_at_[p.first] = ended_at;
      sess_logger->arc2ended_at_[p.second].second = ended_at;
      TimeLinePushBack(p.second, dev);
      AfterRun(p.second);
      PrevArc(p.second->to(), [&](Arc* arc) {
        auto place = dynamic_cast<Node*>(arc);
        auto instance_input = Sess()->batch_arc_mgr().Find(batch, place);
        Sess()->tokens_.erase(instance_input);
      });
      NextArc(p.second->to(), [&](Arc* arc) {
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

}  // namespace schedule
}  // namespace oneflow
