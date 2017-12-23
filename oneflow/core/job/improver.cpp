#include "oneflow/core/job/improver.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/register/register_desc.pb.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

void ParseActEvents(const std::string& act_event_filepath,
                    std::list<ActEvent>* act_events) {
  NormalPersistentInStream in_stream(LocalFS(), act_event_filepath);
  size_t act_event_size;
  while (!in_stream.Read(reinterpret_cast<char*>(&act_event_size),
                         sizeof(size_t))) {
    std::vector<char> buffer(act_event_size);
    CHECK(!in_stream.Read(buffer.data(), act_event_size));
    act_events->emplace_back();
    act_events->back().ParseFromArray(buffer.data(), act_event_size);
  }
}

size_t MemoryConsuming(const std::list<const RegstDescProto*>& regst_descs,
                       const HashMap<int64_t, double>& regst_desc_id2num) {
  size_t mem_consuming = 0;
  for (const RegstDescProto* regst_desc : regst_descs) {
    uint32_t regst_num =
        ceil(regst_desc_id2num.at(regst_desc->regst_desc_id()));
    regst_num = std::max(regst_num,
                         static_cast<uint32_t>(regst_desc->min_register_num()));
    RtRegstDesc runtime_regst_desc(*regst_desc);
    mem_consuming +=
        regst_num * runtime_regst_desc.packed_blob_desc()->TotalByteSize();
  }
  return mem_consuming;
}

void ComputeIIAndRegstDescAvgLifeTime(
    const ActorGraph& graph, double* ii,
    HashMap<int64_t, double>* regst_desc_id2life_time) {
  *ii = graph.InitiationInterval();
  LOG(INFO) << "ii = " << *ii;
  HashMap<int64_t, double> task_id2avg_duration;
  graph.MakeTaskId2AvgDurationHash(&task_id2avg_duration);
  auto AvgDuration4TaskId = [&](int64_t task_id) -> double {
    if (task_id2avg_duration.find(task_id) == task_id2avg_duration.end()) {
      return static_cast<double>(0);
    } else {
      return task_id2avg_duration.at(task_id);
    }
  };
  graph.MakeRegstDescId2AvgLifeTimeHash(regst_desc_id2life_time,
                                        AvgDuration4TaskId);
  //	default value for life_time
  for (auto& pair : *regst_desc_id2life_time) {
    if (pair.second <= 0) { pair.second = *ii; }
    LOG(INFO) << pair.first << "\t" << pair.second;
  }
}

double ComputeLeastIIFlationRatio(
    const std::list<const RegstDescProto*>& regst_descs,
    const HashMap<int64_t, double>& regst_desc_id2num) {
  CHECK(regst_desc_id2num.size());
  double ii_flation_ratio = static_cast<double>(UINT_MAX);
  for (const RegstDescProto* regst_desc : regst_descs) {
    double regst_num = regst_desc_id2num.at(regst_desc->regst_desc_id());
    if (regst_num <= 1) continue;
    double delta = regst_num - floor(regst_num);
    delta = (delta <= 0.000001 ? 1 : delta);
    double ratio = regst_num / (regst_num - delta);
    if (ii_flation_ratio > ratio) { ii_flation_ratio = ratio; }
  }
  return ii_flation_ratio;
}

void SetRegstNum(Plan* plan,
                 const HashMap<int64_t, double>& regst_desc_id2num) {
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      RegstDescProto* regst_desc = &pair.second;
      uint32_t regst_num =
          ceil(regst_desc_id2num.at(regst_desc->regst_desc_id()));
      regst_num = std::max(
          regst_num, static_cast<uint32_t>(regst_desc->min_register_num()));
      regst_desc->set_register_num(regst_num);
      LOG(INFO) << "regst_desc_id: " << regst_desc->regst_desc_id()
                << ", register_num: " << regst_num << std::endl;
    }
  }
}

}  // namespace

size_t Improver::AvailableMemSize(int64_t machine_id,
                                  int64_t memory_zone_id) const {
  return amd_.machine_amd(machine_id).zone_size(memory_zone_id) * 0.95;
}

int64_t Improver::GetMemoryZoneId(const MemoryCase& mem_case) const {
  if (JobDesc::Singleton()->GetDeviceType() == DeviceType::kCPU) { return 0; }
  if (mem_case.has_device_cuda_mem()) {
    return mem_case.device_cuda_mem().device_id();
  }
  return JobDesc::Singleton()->resource().device_num_per_machine();
}

void Improver::MakeMemoryDevice2RegstDescs(
    const Plan& plan,
    std::vector<std::vector<std::list<const RegstDescProto*>>>* mz2regst_desc)
    const {
  for (const auto& task : plan.task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      int64_t mem_zone_id = GetMemoryZoneId(pair.second.mem_case());
      (*mz2regst_desc)[task.machine_id()][mem_zone_id].push_back(&pair.second);
    }
  }
}

bool Improver::IsOutOfMemory(
    int64_t machine_id, int64_t memory_zone_id,
    const std::list<const RegstDescProto*>& regst_descs,
    const HashMap<int64_t, double>& regst_desc_id2num) const {
  return MemoryConsuming(regst_descs, regst_desc_id2num)
         >= AvailableMemSize(machine_id, memory_zone_id);
}

void Improver::FindMinRegstNumWithLeastPerformanceLoss(
    int64_t machine_id, int64_t memory_zone_id, double ii,
    const std::list<const RegstDescProto*>& regst_descs,
    const HashMap<int64_t, double>& regst_desc_id2life_time,
    HashMap<int64_t, double>* regst_desc_id2num) const {
  // shrink memory with least performance loss step by step
  while (IsOutOfMemory(machine_id, memory_zone_id, regst_descs,
                       *regst_desc_id2num)) {
    ii *= ComputeLeastIIFlationRatio(regst_descs, *regst_desc_id2num);
    double max_regst_num = 0;
    for (const RegstDescProto* regst_desc : regst_descs) {
      double rn = regst_desc_id2life_time.at(regst_desc->regst_desc_id()) / ii;
      if (max_regst_num < rn) { max_regst_num = rn; }
      (*regst_desc_id2num)[regst_desc->regst_desc_id()] = rn;
    }
    if (max_regst_num <= 1) { break; }
  }
}

void Improver::MemoryLimitedAllocate(
    const ActorGraph& graph,
    HashMap<int64_t, double>* regst_desc_id2num) const {
  double ii = 0;
  HashMap<int64_t, double> regst_desc_id2life_time;
  ComputeIIAndRegstDescAvgLifeTime(graph, &ii, &regst_desc_id2life_time);
  CHECK(ii > 0);
  for (const auto& pair : regst_desc_id2life_time) {
    (*regst_desc_id2num)[pair.first] = pair.second / ii;
  }
  std::vector<std::vector<std::list<const RegstDescProto*>>> mz_regst_descs;
  MakeMemoryDevice2RegstDescs(graph.plan(), &mz_regst_descs);
  FOR_RANGE(int64_t, machine_id, 0, mz_regst_descs.size()) {
    FOR_RANGE(int64_t, mem_zone_id, 0, mz_regst_descs[machine_id].size()) {
      FindMinRegstNumWithLeastPerformanceLoss(
          machine_id, mem_zone_id, ii, mz_regst_descs[machine_id][mem_zone_id],
          regst_desc_id2life_time, regst_desc_id2num);
    }
  }
}

Plan Improver::Improve(const Plan& naive_plan,
                       const std::string& act_event_filepath) {
  Plan plan(naive_plan);
  auto act_events = of_make_unique<std::list<ActEvent>>();
  ParseActEvents(act_event_filepath, act_events.get());
  ActorGraph actor_graph(plan, std::move(act_events));
  HashMap<int64_t, double> regst_desc_id2num;
  MemoryLimitedAllocate(actor_graph, &regst_desc_id2num);
  SetRegstNum(&plan, regst_desc_id2num);
  return plan;
}

}  // namespace oneflow
