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

double CalcII(double regst_desc_duration, uint64_t regst_num, double ii_scale) {
  return regst_desc_duration / ((regst_num - 1) * ii_scale + 1);
}

uint64_t CalcRegstNum(
    const RegstDescProto& regst_desc,
    const std::function<const HashMap<int64_t, double>&(int64_t)>&
        PathDurations4RegstDescId,
    double ii,
    const std::function<const HashMap<int64_t, double>&(int64_t)>&
        PathIIScales4RegstDescId) {
  int64_t regst_desc_id = regst_desc.regst_desc_id();
  const auto& consumer_actor_id2duration =
      PathDurations4RegstDescId(regst_desc_id);
  const auto& consumer_actor_id2ii_scale =
      PathIIScales4RegstDescId(regst_desc_id);
  uint64_t regst_num = 0;
  for (const auto& pair : consumer_actor_id2duration) {
    double duration = pair.second;
    double ii_scale = consumer_actor_id2ii_scale.at(pair.first);
    uint64_t cur_path_regst_num = ceil(CalcRegstNum(duration, ii, ii_scale));
    regst_num = std::max(regst_num, cur_path_regst_num);
  }
  regst_num =
      std::max(regst_num, static_cast<uint64_t>(regst_desc.min_register_num()));
  regst_num =
      std::min(regst_num, static_cast<uint64_t>(regst_desc.max_register_num()));
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

uint64_t CalcMemoryConsumed(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<const HashMap<int64_t, double>&(int64_t)>&
        PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>&
        PathIIScales4RegstDescId,
    double ii) {
  uint64_t mem_consuming = 0;
  for (const RegstDescProto* regst_desc : regst_descs) {
    uint64_t regst_num = CalcRegstNum(*regst_desc, PathDurations4RegstDescId,
                                      ii, PathIIScales4RegstDescId);
    RtRegstDesc runtime_regst_desc(*regst_desc);
    mem_consuming +=
        regst_num * runtime_regst_desc.packed_blob_desc()->TotalByteSize();
  }
  return mem_consuming;
}

std::function<void(int64_t, uint64_t)> MakeSetterSetPlanRegstNum(Plan* plan) {
  HashMap<int64_t, RegstDescProto*> regst_desc_id2regst_desc;
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      int64_t regst_desc_id = pair.second.regst_desc_id();
      regst_desc_id2regst_desc.insert({regst_desc_id, &pair.second});
    }
  }
  return [regst_desc_id2regst_desc](int64_t regst_desc_id, uint64_t num) {
    RegstDescProto* regst_desc = regst_desc_id2regst_desc.at(regst_desc_id);
    regst_desc->set_register_num(num);
  };
}

std::function<const HashMap<int64_t, double>&(int64_t)>
MakeGetterPathDurations4RegstDescId(const ActGraph& graph) {
  std::shared_ptr<HashMap<int64_t, HashMap<int64_t, double>>>
  regst_desc_id2consumer_id2duration(
      new HashMap<int64_t, HashMap<int64_t, double>>());
  graph.ForEachRegstDescConsumerPathMeanDuration([&](int64_t regst_desc_id,
                                                     int64_t consumer_actor_id,
                                                     double time) {
    (*regst_desc_id2consumer_id2duration)[regst_desc_id][consumer_actor_id] =
        time;
  });
  std::shared_ptr<const HashMap<int64_t, double>> empty(
      new HashMap<int64_t, double>());
  return [regst_desc_id2consumer_id2duration,
          empty](int64_t regst_desc_id) -> const HashMap<int64_t, double>& {
    const auto& it = regst_desc_id2consumer_id2duration->find(regst_desc_id);
    if (it == regst_desc_id2consumer_id2duration->end()) {
      return *empty;
    } else {
      return it->second;
    }
  };
}

double IIScale4Actor(const ActGraph& graph, int64_t consumer_actor_id,
                     double default_ii_scale) {
  if (graph.GetTaskProto(consumer_actor_id).task_type() == TaskType::kMdSave) {
    return Global<JobDesc>::Get()->NumOfBatchesInSnapshot()
           * Global<JobDesc>::Get()->NumOfPiecesInBatch();
  }
  return default_ii_scale;
}

std::function<const HashMap<int64_t, double>&(int64_t)>
MakeGetterPathIIScales4RegstDescId(const ActGraph& graph) {
  std::shared_ptr<HashMap<int64_t, HashMap<int64_t, double>>>
  regst_desc_id2consumer_id2ii_scale(
      new HashMap<int64_t, HashMap<int64_t, double>>());
  graph.ForEachRegstDescConsumerPathIIScale([&](int64_t regst_desc_id,
                                                int64_t consumer_actor_id,
                                                double ii_scale) {
    (*regst_desc_id2consumer_id2ii_scale)[regst_desc_id][consumer_actor_id] =
        IIScale4Actor(graph, consumer_actor_id, ii_scale);
  });
  std::shared_ptr<const HashMap<int64_t, double>> empty(
      new HashMap<int64_t, double>());
  return [regst_desc_id2consumer_id2ii_scale,
          empty](int64_t regst_desc_id) -> const HashMap<int64_t, double>& {
    const auto& it = regst_desc_id2consumer_id2ii_scale->find(regst_desc_id);
    if (it == regst_desc_id2consumer_id2ii_scale->end()) {
      return *empty;
    } else {
      return it->second;
    }
  };
}

}  // namespace

uint64_t Improver::AvailableMemSize(int64_t machine_id,
                                    int64_t memory_zone_id) const {
  int64_t mem_size = amd_.machine_amd(machine_id).zone_size(memory_zone_id);
  JobDesc* job_desc = Global<JobDesc>::Get();
  if (memory_zone_id == job_desc->GpuDeviceNum()) {
    mem_size -= job_desc->reserved_host_mem_byte_size();
    mem_size -= job_desc->persistence_buffer_byte_size()
                * job_desc->PersistenceWorkerNum();
  } else {
    mem_size -= job_desc->reserved_device_mem_byte_size();
  }
  CHECK_GT(mem_size, 0);
  return static_cast<uint64_t>(mem_size);
}

int64_t Improver::GetMemoryZoneId(const MemoryCase& mem_case) const {
  if (mem_case.has_device_cuda_mem()) {
    return mem_case.device_cuda_mem().device_id();
  } else {
    return Global<JobDesc>::Get()->GpuDeviceNum();
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
    const std::function<const HashMap<int64_t, double>&(int64_t)>&
        PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>&
        PathIIScales4RegstDescId,
    double ii) const {
  FOR_RANGE(int64_t, machine_id, 0, mz_regst_descs.size()) {
    FOR_RANGE(int64_t, mem_zone_id, 0, mz_regst_descs[machine_id].size()) {
      const auto& regst_descs = mz_regst_descs[machine_id][mem_zone_id];
      if (CalcMemoryConsumed(regst_descs, PathDurations4RegstDescId,
                             PathIIScales4RegstDescId, ii)
          >= AvailableMemSize(machine_id, mem_zone_id)) {
        return true;
      }
    }
  }
  return false;
}

double Improver::CalcMaxRegstDescDuration(
    const std::function<const HashMap<int64_t, double>&(int64_t)>&
        PathDurations4RegstDescId,
    const MemZoneRegstDescs& mz_regst_descs) const {
  double max_duration = 0;
  for (const auto& zone_regst_descs : mz_regst_descs) {
    for (const auto& regst_descs : zone_regst_descs) {
      for (const RegstDescProto* regst_desc : regst_descs) {
        for (const auto& pair :
             PathDurations4RegstDescId(regst_desc->regst_desc_id())) {
          max_duration = std::max(max_duration, pair.second);
        }
      }
    }
  }
  return max_duration;
}

double Improver::BinarySearchII(
    const std::function<const HashMap<int64_t, double>&(int64_t)>&
        PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>&
        PathIIScales4RegstDescId,
    const MemZoneRegstDescs& mz_regst_descs) const {
  double max_duration =
      CalcMaxRegstDescDuration(PathDurations4RegstDescId, mz_regst_descs);
  CHECK(!IsAnyZoneOutOfMemory(mz_regst_descs, PathDurations4RegstDescId,
                              PathIIScales4RegstDescId, max_duration));
  const double ii_search_threshold = 1;
  double r = max_duration;
  double l = 1.0;
  double mid = 1.0;
  while ((r - l) > ii_search_threshold) {
    mid = (l + r) / 2;
    if (IsAnyZoneOutOfMemory(mz_regst_descs, PathDurations4RegstDescId,
                             PathIIScales4RegstDescId, mid)) {
      l = mid;
    } else {
      r = mid;
    }
  }
  return r;
}

void Improver::MemoryLimitedAllocate(
    const ActGraph& graph,
    const std::function<void(int64_t, uint64_t)>& Handler) const {
  auto PathDurations4RegstDescId = MakeGetterPathDurations4RegstDescId(graph);
  auto PathIIScales4RegstDescId = MakeGetterPathIIScales4RegstDescId(graph);
  MemZoneRegstDescs mz_regst_descs;
  MakeMemZoneRegstDescs(graph.plan(), &mz_regst_descs);
  double ii = BinarySearchII(PathDurations4RegstDescId,
                             PathIIScales4RegstDescId, mz_regst_descs);
  for (const auto& task_proto : graph.plan().task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      uint64_t regst_num = CalcRegstNum(pair.second, PathDurations4RegstDescId,
                                        ii, PathIIScales4RegstDescId);
      Handler(pair.second.regst_desc_id(), regst_num);
    }
  }
}

Plan Improver::Improve(const Plan& naive_plan,
                       const std::string& act_event_filepath) {
  auto act_events = of_make_unique<std::list<ActEvent>>();
  ParseActEvents(act_event_filepath, act_events.get());
  ActGraph act_graph(naive_plan, std::move(act_events));
  Plan plan(naive_plan);
  MemoryLimitedAllocate(act_graph, MakeSetterSetPlanRegstNum(&plan));
  return plan;
}

}  // namespace oneflow
