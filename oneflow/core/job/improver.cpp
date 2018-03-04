#include "oneflow/core/job/improver.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/register/register_desc.pb.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

double CalcRegstNum(double regst_desc_duration, double ii, double ii_scale) {
  return ((ii_scale - 1) * ii + regst_desc_duration) / (ii_scale * ii);
}

double CalcII(double regst_desc_duration, size_t regst_num, double ii_scale) {
  return regst_desc_duration / ((regst_num - 1) * ii_scale + 1);
}

size_t CalcRegstNum(const RegstDescProto& regst_desc,
                    const std::function<double(int64_t)>& Duration4RegstDescId,
                    double ii,
                    const std::function<double(int64_t)>& IIScale4RegstDescId) {
  int64_t regst_desc_id = regst_desc.regst_desc_id();
  double duration = Duration4RegstDescId(regst_desc_id);
  double ratio = IIScale4RegstDescId(regst_desc_id);
  size_t regst_num = ceil(CalcRegstNum(duration, ii, ratio));
  regst_num =
      std::max(regst_num, static_cast<size_t>(regst_desc.min_register_num()));
  regst_num =
      std::min(regst_num, static_cast<size_t>(regst_desc.max_register_num()));
  return regst_num;
}

void ForEachStreamCalcTimePerAct(const std::list<ActEvent>& act_events,
                                 const std::function<void(double)>& Handler) {
  HashMap<int64_t, double> stream_id2time;
  HashMap<int64_t, std::unordered_set<int64_t>> stream_id2act_ids;
  for (const ActEvent& act_event : act_events) {
    auto stream_id = act_event.work_stream_id();
    stream_id2time[stream_id] += Duration4ActEvent(act_event);
    stream_id2act_ids[stream_id].insert(act_event.act_id());
  }
  for (const auto& pair : stream_id2time) {
    Handler(pair.second / stream_id2act_ids.at(pair.first).size());
  }
}

double CalcBaseII(const std::list<ActEvent>& act_events) {
  double initiation_interval = 0;
  ForEachStreamCalcTimePerAct(act_events, [&](double ii) {
    initiation_interval = std::max(initiation_interval, ii);
  });
  return initiation_interval;
}

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

size_t CalcMemoryConsumed(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<double(int64_t)>& Duration4RegstDescId,
    const std::function<double(int64_t)>& IIScale4RegstDescId, double ii) {
  size_t mem_consuming = 0;
  for (const RegstDescProto* regst_desc : regst_descs) {
    size_t regst_num = CalcRegstNum(*regst_desc, Duration4RegstDescId, ii,
                                    IIScale4RegstDescId);
    RtRegstDesc runtime_regst_desc(*regst_desc);
    mem_consuming +=
        regst_num * runtime_regst_desc.packed_blob_desc()->TotalByteSize();
  }
  return mem_consuming;
}

std::vector<double> MakeIISearchSpace(
    const HashMap<int64_t, double>& regst_desc_id2duration,
    const std::function<double(int64_t)>& IIScale4RegstDescId, double base_ii) {
  CHECK_GT(base_ii, 0);
  std::vector<double> search_space;
  for (const auto& pair : regst_desc_id2duration) {
    double ii_ratio = IIScale4RegstDescId(pair.first);
    int max_register_num = ceil(pair.second / base_ii);
    for (int num = 1; num <= max_register_num; ++num) {
      double ii = CalcII(pair.second, num, ii_ratio);
      search_space.push_back(ii);
    }
  }
  std::sort(search_space.begin(), search_space.end());
  return search_space;
}

std::function<void(int64_t, size_t)> MakeSetterSetPlanRegstNum(Plan* plan) {
  HashMap<int64_t, RegstDescProto*> regst_desc_id2regst_desc;
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      int64_t regst_desc_id = pair.second.regst_desc_id();
      regst_desc_id2regst_desc.insert({regst_desc_id, &pair.second});
    }
  }
  return [regst_desc_id2regst_desc](int64_t regst_desc_id, size_t num) {
    RegstDescProto* regst_desc = regst_desc_id2regst_desc.at(regst_desc_id);
    regst_desc->set_register_num(num);
    LOG(INFO) << "regst_desc_id: " << regst_desc->regst_desc_id()
              << ", register_num: " << num;
  };
}

std::function<double(int64_t)> MakeGetterDuration4RegstDescId(
    const ActGraph& graph, HashMap<int64_t, double>* regst_desc_id2duration) {
  graph.ForEachRegstDescMeanDuration([&](int64_t regst_desc_id, double time) {
    regst_desc_id2duration->insert({regst_desc_id, time});
  });
  return [regst_desc_id2duration](int64_t regst_desc_id) {
    const auto& it = regst_desc_id2duration->find(regst_desc_id);
    if (it == regst_desc_id2duration->end()) {
      return static_cast<double>(0);
    } else {
      return it->second;
    }
  };
}

std::function<double(int64_t)> MakeGetterIIScale4RegstDescId(
    const ActGraph& graph) {
  HashMap<int64_t, double> regst_desc_id2ii_ratio;
  graph.ForEachRegstDescIIScale([&](int64_t regst_desc_id, double ratio) {
    regst_desc_id2ii_ratio.insert({regst_desc_id, ratio});
  });
  return [regst_desc_id2ii_ratio](int64_t regst_desc_id) {
    const auto& it = regst_desc_id2ii_ratio.find(regst_desc_id);
    if (it == regst_desc_id2ii_ratio.end()) {
      return static_cast<double>(std::numeric_limits<int64_t>::max());
    } else {
      return it->second;
    }
  };
}

}  // namespace

size_t Improver::AvailableMemSize(int64_t machine_id,
                                  int64_t memory_zone_id) const {
  return amd_.machine_amd(machine_id).zone_size(memory_zone_id) * 0.95;
}

int64_t Improver::GetMemoryZoneId(const MemoryCase& mem_case) const {
  if (mem_case.has_device_cuda_mem()) {
    return mem_case.device_cuda_mem().device_id();
  } else {
    return JobDesc::Singleton()->GpuDeviceNum();
  }
}

void Improver::MakeMemZoneRegstDescs(const Plan& plan,
                                     MemZoneRegstDescs* mz2regst_desc) const {
  mz2regst_desc->resize(amd_.machine_amd_size());
  FOR_RANGE(int64_t, machine_id, 0, amd_.machine_amd_size()) {
    mz2regst_desc->at(machine_id)
        .resize(amd_.machine_amd(machine_id).zone_size_size());
  }
  for (const auto& task : plan.task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      int64_t mem_zone_id = GetMemoryZoneId(pair.second.mem_case());
      mz2regst_desc->at(task.machine_id())
          .at(mem_zone_id)
          .push_back(&pair.second);
    }
  }
}

bool Improver::IsAnyZoneOutOfMemory(
    const MemZoneRegstDescs& mz_regst_descs,
    const std::function<double(int64_t)>& Duration4RegstDescId,
    const std::function<double(int64_t)>& IIScale4RegstDescId,
    double ii) const {
  FOR_RANGE(int64_t, machine_id, 0, mz_regst_descs.size()) {
    FOR_RANGE(int64_t, mem_zone_id, 0, mz_regst_descs[machine_id].size()) {
      const auto& regst_descs = mz_regst_descs[machine_id][mem_zone_id];
      if (CalcMemoryConsumed(regst_descs, Duration4RegstDescId,
                             IIScale4RegstDescId, ii)
          >= AvailableMemSize(machine_id, mem_zone_id)) {
        return true;
      }
    }
  }
  return false;
}

double Improver::BinarySearchII(
    const std::vector<double>& search_space,
    const std::function<double(int64_t)>& Duration4RegstDescId,
    const std::function<double(int64_t)>& IIScale4RegstDescId,
    const MemZoneRegstDescs& mz_regst_descs) const {
  CHECK(!IsAnyZoneOutOfMemory(mz_regst_descs, Duration4RegstDescId,
                              IIScale4RegstDescId,
                              search_space.at(search_space.size() - 1)));
  auto SearchDirection = [&](int index) {
    bool is_cur_ii_ok =
        !IsAnyZoneOutOfMemory(mz_regst_descs, Duration4RegstDescId,
                              IIScale4RegstDescId, search_space.at(index));
    bool is_prev_ii_ok =
        index > 0
        && !IsAnyZoneOutOfMemory(mz_regst_descs, Duration4RegstDescId,
                                 IIScale4RegstDescId,
                                 search_space.at(index - 1));
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
  return search_space.at(mid);
}

void Improver::MemoryLimitedAllocate(
    const ActGraph& graph, double base_ii,
    const std::function<void(int64_t, size_t)>& Handler) const {
  HashMap<int64_t, double> regst_desc_id2duration;
  auto Duration4RegstDescId =
      MakeGetterDuration4RegstDescId(graph, &regst_desc_id2duration);
  auto IIScale4RegstDescId = MakeGetterIIScale4RegstDescId(graph);
  MemZoneRegstDescs mz_regst_descs;
  MakeMemZoneRegstDescs(graph.plan(), &mz_regst_descs);
  const auto& search_space =
      MakeIISearchSpace(regst_desc_id2duration, IIScale4RegstDescId, base_ii);
  double ii = BinarySearchII(search_space, Duration4RegstDescId,
                             IIScale4RegstDescId, mz_regst_descs);
  for (const auto& task_proto : graph.plan().task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      size_t regst_num = CalcRegstNum(pair.second, Duration4RegstDescId, ii,
                                      IIScale4RegstDescId);
      Handler(pair.second.regst_desc_id(), regst_num);
    }
  }
}

Plan Improver::Improve(const Plan& naive_plan,
                       const std::string& act_event_filepath) {
  auto act_events = of_make_unique<std::list<ActEvent>>();
  ParseActEvents(act_event_filepath, act_events.get());
  double base_ii = CalcBaseII(*act_events);
  ActGraph act_graph(naive_plan, std::move(act_events));
  act_graph.ToDotFiles(JoinPath(LogDir(), "/dot/act_sub_graphs"));
  Plan plan(naive_plan);
  MemoryLimitedAllocate(act_graph, base_ii, MakeSetterSetPlanRegstNum(&plan));
  return plan;
}

}  // namespace oneflow
