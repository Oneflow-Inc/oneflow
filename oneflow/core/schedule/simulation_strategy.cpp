/**
 * Copyright 2017 Xinqi Li
 */
#include "oneflow/core/schedule/simulation_strategy.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"

namespace oneflow {
namespace schedule {

void LazyEvaluationStrategy::TimeLinePushBack(TaskInstance* instance,
                                              SDevice* device) {
  schedule_engine()->schedule()->TimeLinePushBack(instance, device);
}

void MemorySimulationStrategy::InitFuncs() {
  get_node_instance_ = [&](TaskArcInstance* arc) {
    auto session = schedule_engine()->session();
    return session->task_instance_mgr().Find(arc->from(), arc->to()->to());
  };

  is_instance_ready_ = std::bind(&MemorySimulationStrategy::IsInstanceReady,
                                 this, std::placeholders::_1);
  get_instance_device_ = std::bind(&SimulatorScheduleEngine::GetInstanceDevice,
                                   schedule_engine(), std::placeholders::_1);
  get_ascendent_ended_at_ =
      std::bind(&MemorySimulationStrategy::GetAscendentEndedAt, this,
                std::placeholders::_1);
  pick_instance_to_run_ = [&](const std::list<TaskInstance*>& instances) {
    return schedule_engine()->PickInstanceToRun(instances);
  };
}

bool MemorySimulationStrategy::IsInstanceReady(TaskInstance* instance) {
  bool ready = true;
  auto session = schedule_engine()->session();
  auto graph = session->graph();
  graph->arc_mgr().InputArc(instance->to(), [&](TaskArc* arc) {
    auto instance_input =
        session->task_arc_instance_mgr().Find(instance->from(), arc);
    if (schedule_engine()->mut_tokens().find(instance_input)
        == schedule_engine()->mut_tokens().end()) {
      ready = false;
    }
  });
  return ready;
}

void LimitedMemoryStrategy::InitFuncIsInstanceReady() {
  is_instance_ready_ = [&](TaskInstance* instance) {
    return IsInstanceReady(instance) && IsAllRegstDescReady(instance);
  };
  get_ascendent_ended_at_ = [&](TaskInstance* instance) {
    auto evaluation = schedule_engine()->evaluation();
    return std::max(evaluation->GetAscendentEndedAt(instance),
                    RegstDescEndedAt(instance));
  };
}

void LazyEvaluationStrategy::Retiming() {
  return schedule_engine()->schedule()->Retiming();
}

void LazyEvaluationStrategy::InitTimeNet() {
  return schedule_engine()->schedule()->InitTimeNet();
}

void LimitedMemoryStrategy::InitRegst(
    const std::function<uint32_t(uint64_t)>& get_regst_num) {
  auto session = schedule_engine()->session();
  auto schedule = schedule_engine()->schedule();
  session->graph()->ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    auto count = get_regst_num(regst_desc->id());
    for (uint32_t i = 0; i < count; i++) {
      auto regst = schedule->mut_regst_node_mgr().Create(
          std::to_string(regst_desc->id()));
      schedule->mut_r2rd_arc_mgr().CreateIfNotFound(regst, regst_desc);
    }
  });
}

float EvaluationSimulationStrategy::GetAscendentEndedAt(
    TaskInstance* instance) {
  float ended_at = 0;
  auto session = schedule_engine()->session();
  auto schedule = schedule_engine()->schedule();
  auto graph = session->graph();
  graph->arc_mgr().Input(instance->to(), [&](STask* node) {
    auto instance_input =
        session->task_instance_mgr().Find(instance->from(), node);
    auto itt = schedule->instance2ended_at().find(instance_input);
    float token_ended_at = INT_MAX;
    if (itt != schedule->instance2ended_at().end()) {
      token_ended_at = itt->second.second;
    }
    ended_at = std::max(ended_at, token_ended_at);
  });
  auto dev = schedule_engine()->GetInstanceDevice(instance);
  return std::max(ended_at, schedule->mut_device2ended_at()[dev]);
}

float MemorySimulationStrategy::GetAscendentEndedAt(TaskInstance* instance) {
  auto evaluation = schedule_engine()->evaluation();
  return evaluation->GetAscendentEndedAt(instance);
}

float LimitedMemoryStrategy::RegstDescEndedAt(TaskInstance* instance) {
  float ended_at = 0;
  auto schedule = schedule_engine()->schedule();
  auto graph = schedule->session()->graph();
  graph->produced_regst_desc_mgr().Output(
      instance->to(), [&](SRegstDesc* regst_desc) {
        auto regst = FindFreeRegst(regst_desc, instance->from());
        ended_at = std::max(ended_at, schedule->mut_regst2ended_at()[regst]);
      });
  return ended_at;
}

void LimitedMemoryStrategy::BeforeRun(TaskInstance* instance) {
  auto session = schedule_engine()->session();
  auto schedule = schedule_engine()->schedule();
  auto graph = schedule->session()->graph();
  graph->produced_regst_desc_mgr().Output(
      instance->to(), [&](SRegstDesc* regst_desc) {
        auto regst = FindFreeRegst(regst_desc, instance->from());
        auto regst_desc_instance = session->regst_desc_instance_mgr().Find(
            instance->from(), regst_desc);
        if (!regst) {
          // BUG
          return;
        }
        schedule->mut_regst_desc_instance2regst()[regst_desc_instance] = regst;
        graph->subscribed_regst_desc_mgr().Input(regst_desc, [&](STask* node) {
          TaskInstance* subscriber_instance =
              session->task_instance_mgr().Find(instance->from(), node);
          schedule->mut_regst_arc_mgr().CreateIfNotFound(subscriber_instance,
                                                         regst);
        });
      });
}

void LimitedMemoryStrategy::AfterRun(TaskInstance* instance) {
  std::list<Arc<TaskInstance, SRegst>*> occupied_arcs;
  auto schedule = schedule_engine()->schedule();
  schedule->regst_arc_mgr().OutputArc(instance, &occupied_arcs);
  for (auto arc : occupied_arcs) {
    schedule->mut_regst2ended_at()[arc->to()] =
        schedule->mut_instance2ended_at()[instance].second;
    schedule->mut_regst_arc_mgr().Delete(arc->id());
  }
}

bool LimitedMemoryStrategy::IsAllRegstDescReady(TaskInstance* instance) {
  bool all_ready = true;
  auto graph = schedule_engine()->session()->graph();
  graph->produced_regst_desc_mgr().Output(
      instance->to(), [&](SRegstDesc* regst_desc) {
        all_ready =
            (all_ready && IsRegstDescReady(regst_desc, instance->from()));
      });
  return all_ready;
}

bool LimitedMemoryStrategy::IsRegstFree(SRegst* regst) {
  auto schedule = schedule_engine()->schedule();
  return schedule->regst_arc_mgr().Input(regst) == 0;
}

bool LimitedMemoryStrategy::IsRegstDescReady(SRegstDesc* regst_desc,
                                             Batch* batch) {
  auto sess = schedule_engine()->session();
  auto schedule = schedule_engine()->schedule();
  auto regst_desc_instance =
      sess->regst_desc_instance_mgr().Find(batch, regst_desc);
  bool free = schedule->mut_regst_desc_instance2regst()[regst_desc_instance];
  if (!free) {
    schedule->r2rd_arc_mgr().Input(regst_desc, [&](SRegst* regst) {
      free = (free || IsRegstFree(regst));
    });
  }
  return free;
}

SRegst* LimitedMemoryStrategy::FindFreeRegst(SRegstDesc* regst_desc,
                                             Batch* batch) {
  auto sess = schedule_engine()->session();
  auto schedule = schedule_engine()->schedule();
  auto regst_desc_instance =
      sess->regst_desc_instance_mgr().Find(batch, regst_desc);
  SRegst* ret = schedule->mut_regst_desc_instance2regst()[regst_desc_instance];
  if (!ret) {
    float ended_at = INT_MAX;
    schedule->r2rd_arc_mgr().Input(regst_desc, [&](SRegst* regst) {
      if (IsRegstFree(regst)) {
        if (schedule->mut_regst2ended_at()[regst] < ended_at) {
          // first recycled register
          ended_at = schedule->mut_regst2ended_at()[regst];
          ret = regst;
        }
      }
    });
  }
  return ret;
}

std::unique_ptr<std::unordered_map<SDevice*, TaskInstance*>>
MemorySimulationStrategy::Pick(std::unordered_set<TaskArcInstance*>* tokens) {
  std::unordered_map<float, std::list<TaskInstance*>>
      instances_groupby_ended_at;
  for (const auto& elem : *tokens) {
    auto node_instance = get_node_instance_(elem);
    auto is_ready = is_instance_ready_(node_instance);
    if (is_ready) {
      auto ended_at = get_ascendent_ended_at_(node_instance);
      instances_groupby_ended_at[ended_at].push_back(node_instance);
    }
  }
  CHECK(instances_groupby_ended_at.size());
  //	pick firstly finished instances
  auto first_finished = instances_groupby_ended_at.begin();
  auto itt = first_finished;
  for (itt++; itt != instances_groupby_ended_at.end(); itt++) {
    if (first_finished->first > itt->first) { first_finished = itt; }
  }
  //	group instances by device
  std::unordered_map<SDevice*, std::list<TaskInstance*>> dev2instances;
  for (const auto& instance : first_finished->second) {
    auto dev = get_instance_device_(instance);
    dev2instances[dev].push_back(instance);
  }
  // 	pick instances to run
  auto instances_picked =
      of_make_unique<std::unordered_map<SDevice*, TaskInstance*>>();
  for (const auto& pair : dev2instances) {
    (*instances_picked)[pair.first] = pick_instance_to_run_(pair.second);
  }
  return instances_picked;
}

}  // namespace schedule
}  // namespace oneflow
