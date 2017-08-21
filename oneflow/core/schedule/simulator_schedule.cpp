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
  auto last = dev2current_instance_[device];
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
      if (next->from() == instance->from()) { cb(next); }
    });
  };

  auto foreach_prev = [&](TaskInstance* instance,
                          const std::function<void(TaskInstance*)>& cb) {
    ForeachPrevTaskInstance(instance, [&](TaskInstance* prev) {
      if (prev->from() == instance->from()) { cb(prev); }
    });
  };

  BfsVisitor<TaskInstance*> bfs_foreach(foreach_next, foreach_prev);

  std::list<STask*> loss_nodes;
  session()->graph()->LossNodes(&loss_nodes);
  auto batch_nodes = session()->GetBatchNodes();
  std::list<TaskInstance*> loss_instances;
  for (Batch* batch : *batch_nodes) {
    for (STask* loss : loss_nodes) {
      auto instance = session()->task_instance_mgr().Find(batch, loss);
      loss_instances.push_back(instance);
    }
  }
  bfs_foreach(loss_instances, cb);
}

void SimulatorSchedule::WalkTimeNetReverse(
    const std::function<void(TaskInstance*)>& cb) {
  auto last_batch = session()->EndBatch();
  auto last_node = session()->graph()->sink();
  auto last_instance =
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
      auto batch = session()->batch_node_mgr().Find(i);
      auto from_node = arc->from();
      auto to_node = arc->to();
      auto from = session()->task_instance_mgr().Find(batch, from_node);
      auto to = session()->task_instance_mgr().Find(batch, to_node);
      mut_timenet_arc_mgr().CreateIfNotFound(from, to);
    }
  });
}

void SimulatorSchedule::Retiming() {
  InitTimeNet();
  float ii = max_interval();
  WalkTimeNetReverse([&](TaskInstance* instance) {
    float lazy_end = INT_MAX;
    uint32_t count =
        timenet_arc_mgr().Output(instance, [&](TaskInstance* instance) {
          const auto& p = mut_instance2ended_at()[instance];
          lazy_end = std::min(lazy_end, p.first);
        });
    auto& p = mut_instance2ended_at()[instance];
    if (!count) { lazy_end = p.second; }
    auto next_instance = session()->GetNextBatchInstance(instance);
    if (next_instance) {
      auto next_instance_end = mut_instance2ended_at()[next_instance].second;
      lazy_end = std::min((float)lazy_end, next_instance_end - ii);
    }
    lazy_end = std::max(lazy_end, p.second);
    auto lazy_start = lazy_end - (p.second - p.first);
    p.second = lazy_end;
    p.first = lazy_start;
  });
  WalkBpTimeNet([&](TaskInstance* instance) {
    float eager_start = 0;
    timenet_arc_mgr().Input(instance, [&](TaskInstance* prev) {
      if (prev->from() == instance->from()) {
        const auto& p = mut_instance2ended_at()[prev];
        eager_start = std::max(eager_start, p.second);
      }
    });
    auto prev_batch_instance = session()->GetPrevBatchInstance(instance);
    if (prev_batch_instance) {
      auto prev_batch_start =
          mut_instance2ended_at()[prev_batch_instance].first;
      eager_start = std::max(eager_start, prev_batch_start + ii);
    }
    auto& p = mut_instance2ended_at()[instance];
    eager_start = std::min(eager_start, p.first);
    auto eager_end = eager_start + (p.second - p.first);
    p.first = eager_start;
    p.second = eager_end;
  });
}

}  // namespace schedule
}  // namespace oneflow
