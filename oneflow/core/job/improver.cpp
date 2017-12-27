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
                       const HashMap<int64_t, double>& regst_desc_id2life_time,
                       double ii) {
  size_t mem_consuming = 0;
  for (const RegstDescProto* regst_desc : regst_descs) {
    uint32_t regst_num =
        ceil(regst_desc_id2life_time.at(regst_desc->regst_desc_id()) / ii);
    regst_num = std::max(regst_num,
                         static_cast<uint32_t>(regst_desc->min_register_num()));
    RtRegstDesc runtime_regst_desc(*regst_desc);
    mem_consuming +=
        regst_num * runtime_regst_desc.packed_blob_desc()->TotalByteSize();
  }
  return mem_consuming;
}

void CalcBaseIIAndRegstDescAvgLifeTime(
    const ActorGraph& graph, double* base_ii,
    HashMap<int64_t, double>* regst_desc_id2life_time) {
  *base_ii = graph.InitiationInterval();
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
    if (pair.second <= 0) { pair.second = *base_ii; }
    LOG(INFO) << "default life time" << pair.first << "\t" << pair.second;
  }
}

std::vector<double> CalcIISearchSpace(HashMap<int64_t, double> id2life_time,
                                      double base_ii) {
  CHECK(base_ii > 0);
  std::vector<double> search_space;
  for (const auto& pair : id2life_time) {
    int max_register_num = ceil(pair.second / base_ii);
    for (int num = 1; num <= max_register_num; ++num) {
      search_space.push_back(pair.second / num);
    }
  }
  std::sort(search_space.begin(), search_space.end());
  return search_space;
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

void Improver::MakeMemZoneRegstDescs(const Plan& plan,
                                     MemZoneRegstDescs* mz2regst_desc) const {
  for (const auto& task : plan.task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      int64_t mem_zone_id = GetMemoryZoneId(pair.second.mem_case());
      (*mz2regst_desc)[task.machine_id()][mem_zone_id].push_back(&pair.second);
    }
  }
}

bool Improver::IsAnyZoneOutOfMemory(
    const MemZoneRegstDescs& mz_regst_descs,
    const HashMap<int64_t, double>& regst_desc_id2life_time, double ii) const {
  FOR_RANGE(int64_t, machine_id, 0, mz_regst_descs.size()) {
    FOR_RANGE(int64_t, mem_zone_id, 0, mz_regst_descs[machine_id].size()) {
      const auto& regst_descs = mz_regst_descs[machine_id][mem_zone_id];
      if (MemoryConsuming(regst_descs, regst_desc_id2life_time, ii)
          >= AvailableMemSize(machine_id, mem_zone_id)) {
        return true;
      }
    }
  }
  return false;
}

double Improver::CalcII(double base_ii,
                        HashMap<int64_t, double> regst_desc_id2life_time,
                        const MemZoneRegstDescs& mz_regst_descs) const {
  auto search_space = CalcIISearchSpace(regst_desc_id2life_time, base_ii);
  CHECK(!IsAnyZoneOutOfMemory(mz_regst_descs, regst_desc_id2life_time,
                              search_space[search_space.size() - 1]));
  auto SearchDirection = [&](int index) {
    bool is_cur_ii_ok = !IsAnyZoneOutOfMemory(
        mz_regst_descs, regst_desc_id2life_time, search_space[index]);
    bool is_prev_ii_ok =
        index > 0
        && !IsAnyZoneOutOfMemory(mz_regst_descs, regst_desc_id2life_time,
                                 search_space[index - 1]);
    if (!is_cur_ii_ok) { return 1; }
    if (is_prev_ii_ok) { return -1; }
    return 0;
  };

  int r = search_space.size() - 1;
  int l = 0;
  int mid = 0;
  while (true) {
    mid = (l + r) / 2;
    auto direction = SearchDirection(mid);
    if (direction == 0) { break; }
    if (direction > 0) {
      l = mid;
    } else {
      r = mid;
    }
  }
  return search_space[mid];
}

void Improver::MemoryLimitedAllocate(
    const ActorGraph& graph,
    HashMap<int64_t, double>* regst_desc_id2num) const {
  double base_ii = 0;
  HashMap<int64_t, double> regst_desc_id2life_time;
  CalcBaseIIAndRegstDescAvgLifeTime(graph, &base_ii, &regst_desc_id2life_time);
  MemZoneRegstDescs mz_regst_descs;
  MakeMemZoneRegstDescs(graph.plan(), &mz_regst_descs);
  double ii = CalcII(base_ii, regst_desc_id2life_time, mz_regst_descs);
  for (const auto& pair : regst_desc_id2life_time) {
    (*regst_desc_id2num)[pair.first] = pair.second / ii;
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
