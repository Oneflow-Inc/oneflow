/**
 * Copyright 2017 Xinqi Li
 */
#include "oneflow/core/schedule/simulator_schedule.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/bfs_visitor.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulation_strategy.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"

namespace oneflow {
namespace schedule {

void SimulatorSchedule::TimeLinePushBack(TaskInstance* instance,
                                         SDevice* device) {
  TaskInstance* last = dev2current_instance_[device];
  if (last) { mut_timenet_arc_mgr().CreateIfNotFound(last, instance); }
  dev2current_instance_[device] = instance;
}

void SimulatorSchedule::WalkBpTimeNet(
    const std::function<void(TaskInstance*)>& cb) {
  //	use split-device hypothesis to avoid complicated
  //	dependences between different batches
  auto foreach_next = [&](TaskInstance* instance,
                          const std::function<void(TaskInstance*)>& cb) {
    ForeachNextTaskInstance(instance, [&](TaskInstance* next) {
      if (next->src_node() == instance->src_node()) { cb(next); }
    });
  };

  auto foreach_prev = [&](TaskInstance* instance,
                          const std::function<void(TaskInstance*)>& cb) {
    ForeachPrevTaskInstance(instance, [&](TaskInstance* prev) {
      if (prev->src_node() == instance->src_node()) { cb(prev); }
    });
  };

  BfsVisitor<TaskInstance*> bfs_foreach(foreach_next, foreach_prev);

  std::list<STask*> loss_nodes;
  session()->graph()->LossNodes(&loss_nodes);
  auto batch_nodes = session()->GetBatchNodes();
  std::list<TaskInstance*> loss_instances;
  for (Batch* batch : *batch_nodes) {
    for (STask* loss : loss_nodes) {
      TaskInstance* instance = session()->task_instance_mgr().Find(batch, loss);
      loss_instances.push_back(instance);
    }
  }
  bfs_foreach(loss_instances, cb);
}

void SimulatorSchedule::WalkTimeNetReverse(
    const std::function<void(TaskInstance*)>& cb) {
  Batch* last_batch = const_cast<Batch*>(session()->EndBatch());
  STask* last_node = session()->graph()->sink();
  TaskInstance* last_instance =
      session()->task_instance_mgr().Find(last_batch, last_node);

  auto foreach_next =
      std::bind(&SimulatorSchedule::ForeachPrevTaskInstance, this,
                std::placeholders::_1, std::placeholders::_2);
  auto foreach_prev =
      std::bind(&SimulatorSchedule::ForeachNextTaskInstance, this,
                std::placeholders::_1, std::placeholders::_2);
  BfsVisitor<TaskInstance*> bfs_foreach(foreach_next, foreach_prev);
  bfs_foreach(last_instance, cb);
}

void SimulatorSchedule::InitTimeNet() {
  session()->graph()->ForeachArc([&](TaskArc* arc) {
    uint32_t start = 0;
    uint32_t end = session()->nr_batch();
    for (uint32_t i = start; i < end; i++) {
      Batch* batch = const_cast<Batch*>(session()->batch_node_mgr().Find(i));
      STask* from_node = arc->src_node();
      STask* to_node = arc->dst_node();
      TaskInstance* src_node =
          session()->task_instance_mgr().Find(batch, from_node);
      TaskInstance* dst_node =
          session()->task_instance_mgr().Find(batch, to_node);
      mut_timenet_arc_mgr().CreateIfNotFound(src_node, dst_node);
    }
  });
}

void SimulatorSchedule::Retiming() {
  InitTimeNet();
  float ii = max_interval();
  std::pair<float, float> defval(0, 0);
  WalkTimeNetReverse([&](TaskInstance* instance) {
    float lazy_end = INT_MAX;
    uint32_t count =
        timenet_arc_mgr().Output(instance, [&](TaskInstance* instance) {
          const auto& p = GetOrDefault(instance2ended_at(), instance, defval);
          lazy_end = std::min(lazy_end, p.first);
        });
    auto& p = mut_instance2ended_at()[instance];
    if (!count) { lazy_end = p.second; }
    TaskInstance* next_instance = session()->GetNextBatchInstance(instance);
    if (next_instance) {
      float next_instance_end =
          GetOrDefault(instance2ended_at(), next_instance, defval).second;
      lazy_end = std::min((float)lazy_end, next_instance_end - ii);
    }
    lazy_end = std::max(lazy_end, p.second);
    float lazy_start = lazy_end - (p.second - p.first);
    p.second = lazy_end;
    p.first = lazy_start;
  });
  WalkBpTimeNet([&](TaskInstance* instance) {
    float eager_start = 0;
    timenet_arc_mgr().Input(instance, [&](TaskInstance* prev) {
      if (prev->src_node() == instance->src_node()) {
        const auto& p = GetOrDefault(instance2ended_at(), prev, defval);
        eager_start = std::max(eager_start, p.second);
      }
    });
    TaskInstance* prev_batch_instance =
        session()->GetPrevBatchInstance(instance);
    if (prev_batch_instance) {
      float prev_batch_start =
          GetOrDefault(instance2ended_at(), prev_batch_instance, defval).first;
      eager_start = std::max(eager_start, prev_batch_start + ii);
    }
    auto& p = mut_instance2ended_at()[instance];
    eager_start = std::min(eager_start, p.first);
    float eager_end = eager_start + (p.second - p.first);
    p.first = eager_start;
    p.second = eager_end;
  });
}

}  // namespace schedule
}  // namespace oneflow
