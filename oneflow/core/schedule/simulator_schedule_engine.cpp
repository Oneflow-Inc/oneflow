/**
 * Copyright 2017 Xinqi Li
 */
#include "oneflow/core/schedule/simulator_schedule_engine.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/bfs_visitor.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulation_strategy.h"

namespace oneflow {
namespace schedule {

void SimulatorSchedule::TimeLinePushBack(TaskInstance* instance,
                                         SDevice* device) {
  auto last = dev2current_instance_[device];
  if (last) { mut_timenet_arc_mgr().CreateIfNotFound(last, instance); }
  dev2current_instance_[device] = instance;
}

void SimulatorSchedule::WalkBpTimeNet(
    SimulatorScheduleEngine* schedule_engine,
    const std::function<void(TaskInstance*)>& cb) {
  CHECK(session() == schedule_engine->session());
  CHECK(this == schedule_engine->schedule());

  //  auto foreach_next =
  //      std::bind(&SimulatorSchedule::ForeachNextTaskInstance, this,
  //                std::placeholders::_1, std::placeholders::_2);
  //  auto foreach_prev =
  //      std::bind(&SimulatorSchedule::ForeachPrevTaskInstance, this,
  //                std::placeholders::_1, std::placeholders::_2);

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
    SimulatorScheduleEngine* schedule_engine,
    const std::function<void(TaskInstance*)>& cb) {
  CHECK(session() == schedule_engine->session());
  CHECK(this == schedule_engine->schedule());
  auto last_batch = schedule_engine->direction_->EndBatch();
  auto last_node = schedule_engine->direction_->EndNode();
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

void SimulatorSchedule::InitTimeNet(SimulatorScheduleEngine* schedule_engine) {
  CHECK(session() == schedule_engine->session());
  CHECK(this == schedule_engine->schedule());
  session()->graph()->ForeachArc([&](TaskArc* arc) {
    uint32_t start = 0;
    uint32_t end = session()->nr_batch();
    for (uint32_t i = start; i < end; i++) {
      auto batch = session()->batch_node_mgr().Find(i);
      auto from_node = schedule_engine->direction_->GetFrom(arc);
      auto to_node = schedule_engine->direction_->GetTo(arc);
      auto from = session()->task_instance_mgr().Find(batch, from_node);
      auto to = session()->task_instance_mgr().Find(batch, to_node);
      mut_timenet_arc_mgr().CreateIfNotFound(from, to);
    }
  });
}

void SimulatorSchedule::Retiming(SimulatorScheduleEngine* schedule_engine) {
  InitTimeNet(schedule_engine);
  CHECK(session() == schedule_engine->session());
  CHECK(this == schedule_engine->schedule());
  float ii = max_interval();
  WalkTimeNetReverse(schedule_engine, [&](TaskInstance* instance) {
    float lazy_end = INT_MAX;
    uint32_t count =
        timenet_arc_mgr().Output(instance, [&](TaskInstance* instance) {
          const auto& p = mut_instance2ended_at()[instance];
          lazy_end = std::min(lazy_end, p.first);
        });
    auto& p = mut_instance2ended_at()[instance];
    if (!count) { lazy_end = p.second; }
    //    auto next_instance = get_next_instance(instance);
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
  WalkBpTimeNet(schedule_engine, [&](TaskInstance* instance) {
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

float SimulatorSchedule::GetDurationByTimeGapToLoss(TaskInstance* from,
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

void SimulatorSchedule::UpdateDuration(
    SimulatorScheduleEngine* schedule_engine) {
  CHECK(session() == schedule_engine->session());
  auto session = schedule_engine->session();
  session->graph()->ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    STask* owner = nullptr;
    session->graph()->produced_regst_desc_mgr().Input(regst_desc, &owner);
    float duration = 0;
    uint32_t start = session->nr_base_batch();
    uint32_t end = start + session->nr_base_batch();
    session->graph()->subscribed_regst_desc_mgr().Input(
        regst_desc, [&](STask* node) {
          float sum = 0;
          std::set<float> cases;
          for (uint32_t i = start; i < end; i++) {
            auto batch = session->batch_node_mgr().Find(i);
            TaskInstance* owner_instance =
                session->task_instance_mgr().Find(batch, owner);
            TaskInstance* node_instance =
                session->task_instance_mgr().Find(batch, node);
            float d = GetDurationByTimeGapToLoss(owner_instance, node_instance);
            cases.insert(d);
          }
          CHECK(cases.size());
          for (float x : cases) { sum += x; }
          float avg = sum / cases.size();
          duration = std::max(duration, avg);
        });
    mut_regst_desc2duration()[regst_desc] = duration;
  });
}

void SimulatorSchedule::UpdateRegstCount() {
  session()->graph()->ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    STask* owner = nullptr;
    session()->graph()->produced_regst_desc_mgr().Input(regst_desc, &owner);
    auto duration = mut_regst_desc2duration()[regst_desc];
    auto interval = max_interval();
    uint32_t count = ceil(duration / std::max(interval, 1.0f));
    count = std::max(count, regst_desc->min_regst_count());
    mut_regst_desc2count()[regst_desc] = count;
    std::cout << "Allocation\t" << regst_desc->id() << "\t" << count << "\t"
              << duration << "," << interval << std::endl;
  });
}

void SimulatorSchedule::UpdateInterval(
    SimulatorScheduleEngine* schedule_engine) {
  auto session = schedule_engine->session();
  STask* end_node = schedule_engine->EndNode();
  float sum = 0.0;
  float last_time = 0.0;
  uint32_t start = session->nr_base_batch();
  uint32_t end = start + session->nr_base_batch();
  CHECK(end - start > 1);
  std::set<float> cases;
  for (uint32_t i = start; i < end; i++) {
    auto batch = session->batch_node_mgr().Find(i);
    auto instance = session->task_instance_mgr().Find(batch, end_node);
    auto start_time =
        schedule_engine->GetTime(mut_instance2ended_at()[instance].first);
    if (i > start) { cases.insert(start_time - last_time); }
    last_time = start_time;
  }
  for (float x : cases) { sum += x; }
  mut_max_interval() = sum / cases.size();
}

void SimulatorScheduleEngine::ClearTmpData() {
  tokens_.clear();
  schedule()->Clear();
}

void SimulatorSchedule::Clear() {
  mut_instance2ended_at().clear();
  mut_device2ended_at().clear();
  mut_regst_desc2duration().clear();
}

void SimulatorSchedule::MergeTimeGapToLossInPlace(SimulatorSchedule* schedule) {
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
  merge(&mut_start_time_gap_to_loss(), &schedule->mut_start_time_gap_to_loss());
  merge(&mut_end_time_gap_to_loss(), &schedule->mut_end_time_gap_to_loss());
}

void SimulatorSchedule::UpdateTimeGapToLoss(
    SimulatorScheduleEngine* schedule_engine) {
  auto session = schedule_engine->session();
  std::list<STask*> loss_nodes;
  session->graph()->LossNodes(&loss_nodes);
  uint32_t start = 0;
  uint32_t end = start + session->nr_batch();
  for (uint32_t i = start; i < end; i++) {
    auto batch = session->batch_node_mgr().Find(i);
    for (STask* loss : loss_nodes) {
      auto loss_instance = session->task_instance_mgr().Find(batch, loss);
      auto loss_start_time =
          schedule_engine->GetStartTime(mut_instance2ended_at()[loss_instance]);
      auto loss_end_time =
          schedule_engine->GetEndTime(mut_instance2ended_at()[loss_instance]);
      float loss_middle_time =
          ((float)loss_start_time + (float)loss_end_time) / 2;
      auto set_time_gap = [&](STask* node) {
        auto node_instance = session->task_instance_mgr().Find(batch, node);
        float start_time = schedule_engine->GetStartTime(
            mut_instance2ended_at()[node_instance]);
        float end_time =
            schedule_engine->GetEndTime(mut_instance2ended_at()[node_instance]);
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

void SimulatorScheduleEngine::NewSinkTokens() {
  ClearTmpData();
  std::list<TaskArc*> arcs;
  auto graph = session()->graph();
  graph->arc_mgr().InputArc(graph->sink(), &arcs);
  auto batchs = session()->GetBatchNodes();
  session()->task_arc_instance_mgr().Find(
      *batchs, arcs,
      [&](TaskArcInstance* instance) { tokens_.insert(instance); });
  InitNodeBatchInstance(graph->sink());
}

void SimulatorScheduleEngine::InitNodeBatchInstance(STask* node) {
  for (uint32_t i = 0; i < session()->nr_batch(); i++) {
    auto batch = session()->batch_node_mgr().Find(i);
    auto start_instance = session()->mut_task_instance_mgr().Find(batch, node);
    schedule()->mut_instance2ended_at()[start_instance] =
        std::make_pair(0u, 0u);
  }
}

void SimulatorScheduleEngine::NewSourceTokens() {
  ClearTmpData();
  std::list<TaskArc*> arcs;
  auto graph = session()->graph();
  graph->arc_mgr().OutputArc(graph->source(), &arcs);
  auto batchs = session()->GetBatchNodes();
  session()->task_arc_instance_mgr().Find(
      *batchs, arcs,
      [&](TaskArcInstance* instance) { tokens_.insert(instance); });
  InitNodeBatchInstance(graph->source());
}

SDevice* SimulatorScheduleEngine::GetInstanceDevice(TaskInstance* instance) {
  SDevice* ret = nullptr;
  session()->graph()->device_arc_mgr().Output(instance->to(), &ret);
  return ret;
}

void SimulatorScheduleEngine::InitStrategies() {
  SetStrategy(unique_ptr_new<PositiveDirectionStrategy>(this));
  SetStrategy(unique_ptr_new<LazyEvaluationStrategy>(this));
  SetStrategy(unique_ptr_new<LimitedMemoryStrategy>(this));
}

std::unique_ptr<Schedule> SimulatorScheduleEngine::RunInTwoDirections(
    const std::function<uint32_t(uint64_t)>& get_regst_num) {
  SetStrategy(unique_ptr_new<PositiveDirectionStrategy>(this));
  auto positive_schedule = Run(get_regst_num);
  positive_schedule->UpdateDuration(this);
  positive_schedule->UpdateRegstCount();
  //  SetStrategy(unique_ptr_new<NegativeDirectionStrategy>(this));
  //  auto negative_schedule = Run(get_regst_num);
  //  negative_schedule->UpdateDuration(this);
  //  negative_schedule->UpdateRegstCount();
  //  positive_schedule->MergeTimeGapToLossInPlace(negative_schedule.get());
  //  positive_schedule->UpdateDuration(this);
  //  positive_schedule->UpdateRegstCount();
  return std::move(positive_schedule);
}

std::unique_ptr<Schedule> SimulatorScheduleEngine::StaticSchedule(
    const std::function<uint32_t(uint64_t)>& get_regst_num) {
  SetStrategy(unique_ptr_new<LimitedMemoryStrategy>(this));
  return RunInTwoDirections(get_regst_num);
}

std::unique_ptr<Schedule> SimulatorScheduleEngine::StaticSchedule() {
  SetStrategy(unique_ptr_new<UnlimitedMemoryStrategy>(this));
  return RunInTwoDirections([](uint64_t) { return static_cast<uint32_t>(2u); });
}

std::unique_ptr<SimulatorSchedule> SimulatorScheduleEngine::Run(
    const std::function<uint32_t(uint64_t)>& get_regst_num) {
  InitRegst(get_regst_num);
  NewStartTokens();
  while (mut_tokens().size()) {
    auto instances_picked = Pick(&mut_tokens());
    for (const auto& p : *instances_picked) {
      auto dev = dynamic_cast<SDevice*>(p.first);
      auto batch = p.second->from();
      BeforeRun(p.second);
      float ended_at = GetAscendentEndedAt(p.second);
      schedule()->mut_instance2ended_at()[p.second].first = ended_at;
      ended_at += (dev ? dev->time() : 0);
      schedule()->mut_device2ended_at()[p.first] = ended_at;
      schedule()->mut_instance2ended_at()[p.second].second = ended_at;
      TimeLinePushBack(p.second, dev);
      AfterRun(p.second);
      PrevArc(p.second->to(), [&](TaskArc* arc) {
        auto instance_input =
            session()->task_arc_instance_mgr().Find(batch, arc);
        mut_tokens().erase(instance_input);
      });
      NextArc(p.second->to(), [&](TaskArc* arc) {
        auto instance_output =
            session()->task_arc_instance_mgr().Find(batch, arc);
        mut_tokens().insert(instance_output);
      });
    }
    if (!instances_picked->size()) { break; }
  }
  schedule()->UpdateInterval(this);
  Retiming();
  schedule()->UpdateTimeGapToLoss(this);
  schedule()->UpdateDuration(this);
  return GetSchedule();
}

}  // namespace schedule
}  // namespace oneflow
