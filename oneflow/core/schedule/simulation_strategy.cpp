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
    const Session& session = schedule_engine()->session();
    return session.task_instance_mgr().Find(arc->src_node(),
                                            arc->dst_node()->dst_node());
  };

  is_instance_ready_ = std::bind(&MemorySimulationStrategy::IsInstanceReady,
                                 this, std::placeholders::_1);
  get_instance_device_ = std::bind(&SimulatorScheduleEngine::GetInstanceDevice,
                                   schedule_engine(), std::placeholders::_1);
  get_ascendant_ended_at_ =
      std::bind(&MemorySimulationStrategy::GetAscendantEndedAt, this,
                std::placeholders::_1);
  pick_instance_to_run_ = [&](const std::list<TaskInstance*>& instances) {
    return schedule_engine()->PickInstanceToRun(instances);
  };
}

bool MemorySimulationStrategy::IsInstanceReady(TaskInstance* instance) {
  bool ready = true;
  const Session& sess = schedule_engine()->session();
  const SGraph& graph = sess.sgraph();
  graph.arc_mgr().InputArc(instance->dst_node(), [&](TaskArc* arc) {
    TaskArcInstance* instance_input =
        sess.task_arc_instance_mgr().Find(instance->src_node(), arc);
    if (schedule_engine()->tokens().find(instance_input)
        == schedule_engine()->tokens().end()) {
      ready = false;
    }
  });
  return ready;
}

void LimitedMemoryStrategy::InitFuncIsInstanceReady() {
  is_instance_ready_ = [&](TaskInstance* instance) {
    return IsInstanceReady(instance) && IsAllRegstDescReady(instance);
  };
  get_ascendant_ended_at_ = [&](TaskInstance* instance) {
    EvaluationSimulationStrategy* evaluation = schedule_engine()->evaluation();
    return std::max(evaluation->GetAscendantEndedAt(instance),
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
  SimulatorSchedule* schedule = schedule_engine()->schedule();
  schedule_engine()->sgraph().ForeachRegstDesc([&](SRegstDesc* regst_desc) {
    uint32_t count = get_regst_num(regst_desc->id());
    for (uint32_t i = 0; i < count; i++) {
      SRegst* regst = schedule->mut_regst_node_mgr()->Create(
          std::to_string(regst_desc->id()));
      schedule->mut_r2rd_arc_mgr()->CreateIfNotFound(regst, regst_desc);
    }
  });
}

float EvaluationSimulationStrategy::GetAscendantEndedAt(
    TaskInstance* instance) {
  float ended_at = 0;
  const Session& session = schedule_engine()->session();
  SimulatorSchedule* schedule = schedule_engine()->schedule();
  session.sgraph().arc_mgr().Input(instance->dst_node(), [&](STask* node) {
    TaskInstance* instance_input =
        session.task_instance_mgr().Find(instance->src_node(), node);
    auto itt = schedule->instance2ended_at().find(instance_input);
    float token_ended_at = INT_MAX;
    if (itt != schedule->instance2ended_at().end()) {
      token_ended_at = itt->second.second;
    }
    ended_at = std::max(ended_at, token_ended_at);
  });
  SDevice* dev = schedule_engine()->GetInstanceDevice(instance);
  float dev_ended_at =
      GetOrDefault(schedule->device2ended_at(), dev, static_cast<float>(0));
  return std::max(ended_at, dev_ended_at);
}

float MemorySimulationStrategy::GetAscendantEndedAt(TaskInstance* instance) {
  EvaluationSimulationStrategy* evaluation = schedule_engine()->evaluation();
  return evaluation->GetAscendantEndedAt(instance);
}

float LimitedMemoryStrategy::RegstDescEndedAt(TaskInstance* instance) {
  float ended_at = 0;
  SimulatorSchedule* schedule = schedule_engine()->schedule();
  schedule_engine()->sgraph().produced_regst_desc_mgr().Output(
      instance->dst_node(), [&](SRegstDesc* regst_desc) {
        SRegst* regst = FindFreeRegst(regst_desc, instance->src_node());
        float regst_ended_at = GetOrDefault(schedule->regst2ended_at(), regst,
                                            static_cast<float>(0));
        ended_at = std::max(ended_at, regst_ended_at);
      });
  return ended_at;
}

void LimitedMemoryStrategy::BeforeRun(TaskInstance* instance, float time) {
  const Session& session = schedule_engine()->session();
  SimulatorSchedule* schedule = schedule_engine()->schedule();
  const SGraph& graph = schedule_engine()->sgraph();
  graph.produced_regst_desc_mgr().Output(
      instance->dst_node(), [&](SRegstDesc* regst_desc) {
        SRegst* regst = FindFreeRegst(regst_desc, instance->src_node());
        RegstDescInstance* regst_desc_instance =
            session.regst_desc_instance_mgr().Find(instance->src_node(),
                                                   regst_desc);
        if (!regst) { return; }
        schedule->mut_regst_desc_instance2regst()[regst_desc_instance] = regst;
        graph.subscribed_regst_desc_mgr().Input(regst_desc, [&](STask* node) {
          TaskInstance* subscriber_instance =
              session.task_instance_mgr().Find(instance->src_node(), node);
          schedule->mut_regst_arc_mgr()->CreateIfNotFound(subscriber_instance,
                                                          regst);
        });
      });
  schedule->EmitBeforeRunEvent(instance, time);
}

void LimitedMemoryStrategy::AfterRun(TaskInstance* instance, float time) {
  std::list<Arc<TaskInstance, SRegst>*> occupied_arcs;
  SimulatorSchedule* schedule = schedule_engine()->schedule();
  schedule->EmitAfterRunEvent(instance, time);
  schedule->regst_arc_mgr().OutputArc(instance, &occupied_arcs);
  std::pair<float, float> zero_range;
  for (Arc<TaskInstance, SRegst>* arc : occupied_arcs) {
    schedule->mut_regst2ended_at()[arc->dst_node()] =
        GetOrDefault(schedule->instance2ended_at(), instance, zero_range)
            .second;
    schedule->mut_regst_arc_mgr()->Delete(arc->id());
  }
}

bool LimitedMemoryStrategy::IsAllRegstDescReady(TaskInstance* instance) {
  bool all_ready = true;
  schedule_engine()->sgraph().produced_regst_desc_mgr().Output(
      instance->dst_node(), [&](SRegstDesc* regst_desc) {
        all_ready =
            (all_ready && IsRegstDescReady(regst_desc, instance->src_node()));
      });
  return all_ready;
}

bool LimitedMemoryStrategy::IsRegstFree(SRegst* regst) {
  SimulatorSchedule* schedule = schedule_engine()->schedule();
  return schedule->regst_arc_mgr().Input(regst) == 0;
}

bool LimitedMemoryStrategy::IsRegstDescReady(SRegstDesc* regst_desc,
                                             Batch* batch) {
  const Session& sess = schedule_engine()->session();
  SimulatorSchedule* schedule = schedule_engine()->schedule();
  RegstDescInstance* regst_desc_instance =
      sess.regst_desc_instance_mgr().Find(batch, regst_desc);
  bool free = GetOrDefault(schedule->regst_desc_instance2regst(),
                           regst_desc_instance, static_cast<SRegst*>(nullptr));
  if (!free) {
    schedule->r2rd_arc_mgr().Input(regst_desc, [&](SRegst* regst) {
      free = (free || IsRegstFree(regst));
    });
  }
  return free;
}

SRegst* LimitedMemoryStrategy::FindFreeRegst(SRegstDesc* regst_desc,
                                             Batch* batch) {
  const Session& sess = schedule_engine()->session();
  SimulatorSchedule* schedule = schedule_engine()->schedule();
  RegstDescInstance* regst_desc_instance =
      sess.regst_desc_instance_mgr().Find(batch, regst_desc);
  SRegst* ret =
      GetOrDefault(schedule->regst_desc_instance2regst(), regst_desc_instance,
                   static_cast<SRegst*>(nullptr));
  float defval = 0;
  if (!ret) {
    float ended_at = INT_MAX;
    schedule->r2rd_arc_mgr().Input(regst_desc, [&](SRegst* regst) {
      if (IsRegstFree(regst)) {
        if (GetOrDefault(schedule->regst2ended_at(), regst, defval)
            < ended_at) {
          // first recycled register
          ended_at = GetOrDefault(schedule->regst2ended_at(), regst, defval);
          ret = regst;
        }
      }
    });
  }
  return ret;
}

std::unique_ptr<std::unordered_map<SDevice*, TaskInstance*>>
MemorySimulationStrategy::Pick(
    const std::unordered_set<TaskArcInstance*>& tokens) {
  std::unordered_map<float, std::list<TaskInstance*>>
      instances_groupby_ended_at;
  for (const auto& elem : tokens) {
    TaskInstance* node_instance = get_node_instance_(elem);
    bool is_ready = is_instance_ready_(node_instance);
    if (is_ready) {
      float ended_at = get_ascendant_ended_at_(node_instance);
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
    SDevice* dev = get_instance_device_(instance);
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
