#include "oneflow/core/job/improver.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/register/register_desc.pb.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/profiler.h"

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
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    double ii,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId) {
  int64_t regst_desc_id = regst_desc.regst_desc_id();
  const auto& consumer_actor_id2duration = PathDurations4RegstDescId(regst_desc_id);
  const auto& consumer_actor_id2ii_scale = PathIIScales4RegstDescId(regst_desc_id);
  uint64_t regst_num = 0;
  for (const auto& pair : consumer_actor_id2duration) {
    double duration = pair.second;
    double ii_scale = consumer_actor_id2ii_scale.at(pair.first);
    uint64_t cur_path_regst_num = ceil(CalcRegstNum(duration, ii, ii_scale));
    regst_num = std::max(regst_num, cur_path_regst_num);
  }
  regst_num = std::max(regst_num, static_cast<uint64_t>(regst_desc.min_register_num()));
  regst_num = std::min(regst_num, static_cast<uint64_t>(regst_desc.max_register_num()));
  return regst_num;
}

void ParseActEvents(const std::string& act_event_filepath, std::list<ActEvent>* act_events) {
  NormalPersistentInStream in_stream(LocalFS(), act_event_filepath);
  int64_t act_event_size;
  while (!in_stream.Read(reinterpret_cast<char*>(&act_event_size), sizeof(act_event_size))) {
    std::vector<char> buffer(act_event_size);
    CHECK(!in_stream.Read(buffer.data(), act_event_size));
    act_events->emplace_back();
    act_events->back().ParseFromArray(buffer.data(), act_event_size);
  }
}

uint64_t CalcMemoryConsumed(
    const std::list<const RegstDescProto*>& regst_descs,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
    double ii) {
  uint64_t mem_consuming = 0;
  for (const RegstDescProto* regst_desc : regst_descs) {
    uint64_t regst_num =
        CalcRegstNum(*regst_desc, PathDurations4RegstDescId, ii, PathIIScales4RegstDescId);
    RtRegstDesc runtime_regst_desc(*regst_desc);
    mem_consuming += regst_num * runtime_regst_desc.packed_blob_desc()->TotalByteSize();
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

std::function<const HashMap<int64_t, double>&(int64_t)> MakeGetterPathDurations4RegstDescId(
    const ActGraph& graph) {
  std::shared_ptr<HashMap<int64_t, HashMap<int64_t, double>>> regst_desc_id2consumer_id2duration(
      new HashMap<int64_t, HashMap<int64_t, double>>());
  graph.ForEachRegstDescConsumerPathMeanDuration(
      [&](int64_t regst_desc_id, int64_t consumer_actor_id, double time) {
        (*regst_desc_id2consumer_id2duration)[regst_desc_id][consumer_actor_id] = time;
      });
  std::shared_ptr<const HashMap<int64_t, double>> empty(new HashMap<int64_t, double>());
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

uint64_t NumOfPiecesInSnapshot() {
  return Global<JobDesc>::Get()->NumOfBatchesInSnapshot()
         * Global<JobDesc>::Get()->NumOfPiecesInBatch();
}

double FormalDuration4ExperimentalDuration(TaskType task_type, double duration,
                                           double act_frequency) {
  if (task_type == TaskType::kMdSave) {
    double formal_run_frequency = 1.0 / NumOfPiecesInSnapshot();
    return (duration / act_frequency) * formal_run_frequency;
  }
  return duration;
}

double CalcBaseII(const ActGraph& act_graph) {
  int64_t max_act_cnt = 0;
  for (const auto& pair : act_graph.actor_id2act_cnt()) {
    if (max_act_cnt < pair.second) { max_act_cnt = pair.second; }
  }
  HashMap<int64_t, double> actor_id2act_frequency;
  for (const auto& pair : act_graph.actor_id2act_cnt()) {
    actor_id2act_frequency[pair.first] = 1.0 * pair.second / max_act_cnt;
  }
  HashMap<int64_t, double> stream_id2total_calc_time;
  act_graph.ForEachNode([&](const ActNode* act_node) {
    int64_t stream_id = act_node->act_event().work_stream_id();
    int64_t actor_id = act_node->actor_id();
    TaskType task_type = act_graph.GetTaskProto(actor_id).task_type();
    stream_id2total_calc_time[stream_id] += FormalDuration4ExperimentalDuration(
        task_type, act_node->Duration(), actor_id2act_frequency.at(actor_id));
  });
  double base_ii = 0;
  for (const auto& pair : stream_id2total_calc_time) {
    base_ii = std::max(base_ii, pair.second / max_act_cnt);
  }
  return base_ii;
}

double IIScale4Actor(TaskType task_type, double default_ii_scale) {
  if (task_type == TaskType::kMdSave) { return NumOfPiecesInSnapshot(); }
  return default_ii_scale;
}

void PushAvgActTimeToProfiler(const ActGraph& act_graph) {
  for (const auto& pair : act_graph.actor_id2total_act_time()) {
    double act_time = pair.second / act_graph.actor_id2act_cnt().at(pair.first);
    Global<Profiler>::Get()->PushAvgActTime(pair.first, act_time);
  }
}

std::function<const HashMap<int64_t, double>&(int64_t)> MakeGetterPathIIScales4RegstDescId(
    const ActGraph& graph) {
  std::shared_ptr<HashMap<int64_t, HashMap<int64_t, double>>> regst_desc_id2consumer_id2ii_scale(
      new HashMap<int64_t, HashMap<int64_t, double>>());
  graph.ForEachRegstDescConsumerPathIIScale(
      [&](int64_t regst_desc_id, int64_t consumer_actor_id, double ii_scale) {
        TaskType task_type = graph.GetTaskProto(consumer_actor_id).task_type();
        (*regst_desc_id2consumer_id2ii_scale)[regst_desc_id][consumer_actor_id] =
            IIScale4Actor(task_type, ii_scale);
      });
  std::shared_ptr<const HashMap<int64_t, double>> empty(new HashMap<int64_t, double>());
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

std::function<void(const std::vector<int64_t>&)> MakeSetterAddCtrlRegst(Plan* plan) {
  HashMap<int64_t, TaskProto*> task_id2task_proto;
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task_proto = plan->mutable_task(i);
    CHECK(task_id2task_proto.insert({task_proto->task_id(), task_proto}).second);
  }
  return [&task_id2task_proto](const std::vector<int64_t>& shared_mem_task_ids) {
    if (shared_mem_task_ids.size() < 2) { return; }
    int64_t first_task_id = shared_mem_task_ids[0];
    int64_t second_task_id = shared_mem_task_ids[1];
    int64_t last_task_id = shared_mem_task_ids.back();
    TaskProto* header_task_proto = task_id2task_proto[first_task_id];
    int64_t shared_regst_desc_id = -1;
    CHECK_NOTNULL(header_task_proto);
    for (const auto& pair : header_task_proto->produced_regst_desc()) {
      for (int64_t consumer_task_id : pair.second.consumer_task_id()) {
        if (second_task_id == consumer_task_id) {
          shared_regst_desc_id = pair.second.regst_desc_id();
        }
      }
    }
    TaskProto* last_task_proto = task_id2task_proto[last_task_id];
    CHECK_NOTNULL(last_task_proto);
    for (const auto& pair : last_task_proto->produced_regst_desc()) {
      if (pair.second.regst_desc_id() == shared_regst_desc_id) {
        CHECK_GT(pair.second.consumer_task_id_size(), 0);
        int64_t one_consumer = pair.second.consumer_task_id(0);
        int64_t ctrl_regst_desc_id = Global<IDMgr>::Get()->NewRegstDescId();
        RegstDescProto ctrl_regst_proto;
        ctrl_regst_proto.set_regst_desc_id(ctrl_regst_desc_id);
        ctrl_regst_proto.set_producer_task_id(first_task_id);
        ctrl_regst_proto.add_consumer_task_id(one_consumer);
        ctrl_regst_proto.set_min_register_num(1);
        ctrl_regst_proto.set_max_register_num(1);
        ctrl_regst_proto.set_register_num(1);
        ctrl_regst_proto.mutable_regst_desc_type()->mutable_delay_regst_desc();
        if (Global<IDMgr>::Get()->GetDeviceTypeFromActorId(first_task_id) == DeviceType::kCPU) {
          ctrl_regst_proto.mutable_mem_case()->mutable_host_mem();
        } else if (Global<IDMgr>::Get()->GetDeviceTypeFromActorId(first_task_id)
                   == DeviceType::kGPU) {
          ctrl_regst_proto.mutable_mem_case()->mutable_device_cuda_mem()->set_device_id(
              Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(header_task_proto->thrd_id()));
        } else {
          UNIMPLEMENTED();
        }
        std::string ctrl_regst_name = "shared_mem_regst_guard_" + std::to_string(first_task_id)
                                      + "_to_" + std::to_string(one_consumer);
        CHECK(header_task_proto->mutable_produced_regst_desc()
                  ->insert({ctrl_regst_name, ctrl_regst_proto})
                  .second);
        TaskProto* sink_task_proto = task_id2task_proto[one_consumer];
        CHECK_NOTNULL(sink_task_proto);
        RegstDescIdSet sink_regst_desc_id_set;
        sink_regst_desc_id_set.add_regst_desc_id(ctrl_regst_desc_id);
        CHECK(sink_task_proto->mutable_consumed_regst_desc_id()
                  ->insert({ctrl_regst_name, sink_regst_desc_id_set})
                  .second);
      }
    }
  };
}

void ForEachMemSharingCriticalSection(
    const Plan& plan, const std::function<void(const std::vector<int64_t>&)>& Handler) {
  HashMap<int32_t, std::vector<std::pair<int64_t, int32_t>>> mem_sharing_id2task_ids;
  for (int i = 0; i < plan.task_size(); i++) {
    for (const auto& pair : plan.task(i).produced_regst_desc()) {
      if (pair.second.has_mem_sharing_info()) {
        int64_t regst_desc_id = pair.second.regst_desc_id();
        int32_t mem_sharing_id = pair.second.mem_sharing_info().mem_shared_id();
        int32_t mem_sharing_order = pair.second.mem_sharing_info().used_order_value();
        CHECK_GE(regst_desc_id, 0);
        CHECK_GE(mem_sharing_order, 0);
        mem_sharing_id2task_ids[mem_sharing_id].push_back({regst_desc_id, mem_sharing_order});
      }
    }
  }
  for (auto& pair : mem_sharing_id2task_ids) {
    std::sort(pair.second.begin(), pair.second.end(),
              [](const std::pair<int64_t, int32_t>& lhs, const std::pair<int64_t, int32_t>& rhs) {
                CHECK_NE(lhs.second, rhs.second);
                return lhs.second < rhs.second;
              });
    std::vector<int64_t> task_ids;
    std::transform(pair.second.begin(), pair.second.end(), task_ids.begin(),
                   [](const std::pair<int64_t, int32_t>& regst_desc_id2mem_sharing_order) {
                     return regst_desc_id2mem_sharing_order.first;
                   });
    Handler(task_ids);
  }
}

}  // namespace

Plan Improver::AddCtrlRegstForMemSharingCriticalSection(const Plan& plan) const {
  Plan ret(plan);
  ForEachMemSharingCriticalSection(plan, MakeSetterAddCtrlRegst(&ret));
  return ret;
}

uint64_t Improver::AvailableMemSize(int64_t machine_id, int64_t memory_zone_id) const {
  int64_t mem_size = amd_.machine_amd(machine_id).zone_size(memory_zone_id);
  JobDesc* job_desc = Global<JobDesc>::Get();
  if (memory_zone_id == job_desc->GpuDeviceNum()) {
    mem_size -= job_desc->reserved_host_mem_byte();
    mem_size -= job_desc->persistence_buf_byte() * record_load_task_num_.at(machine_id);
  } else {
    mem_size -= job_desc->reserved_device_mem_byte();
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

void Improver::MakeMemZoneRegstDescs(const Plan& plan, MemZoneRegstDescs* mz2regst_desc) const {
  mz2regst_desc->resize(amd_.machine_amd_size());
  FOR_RANGE(int64_t, machine_id, 0, amd_.machine_amd_size()) {
    mz2regst_desc->at(machine_id).resize(amd_.machine_amd(machine_id).zone_size_size());
  }
  for (const auto& task : plan.task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      int64_t mem_zone_id = GetMemoryZoneId(pair.second.mem_case());
      mz2regst_desc->at(task.machine_id()).at(mem_zone_id).push_back(&pair.second);
    }
  }
}

bool Improver::IsAnyZoneOutOfMemory(
    const MemZoneRegstDescs& mz_regst_descs,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
    double ii) const {
  FOR_RANGE(int64_t, machine_id, 0, mz_regst_descs.size()) {
    FOR_RANGE(int64_t, mem_zone_id, 0, mz_regst_descs[machine_id].size()) {
      const auto& regst_descs = mz_regst_descs[machine_id][mem_zone_id];
      if (CalcMemoryConsumed(regst_descs, PathDurations4RegstDescId, PathIIScales4RegstDescId, ii)
          >= AvailableMemSize(machine_id, mem_zone_id)) {
        return true;
      }
    }
  }
  return false;
}

double Improver::CalcMaxRegstDescDuration(
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const MemZoneRegstDescs& mz_regst_descs) const {
  double max_duration = 0;
  for (const auto& zone_regst_descs : mz_regst_descs) {
    for (const auto& regst_descs : zone_regst_descs) {
      for (const RegstDescProto* regst_desc : regst_descs) {
        for (const auto& pair : PathDurations4RegstDescId(regst_desc->regst_desc_id())) {
          max_duration = std::max(max_duration, pair.second);
        }
      }
    }
  }
  return max_duration;
}

double Improver::BinarySearchII(
    double base_ii,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathDurations4RegstDescId,
    const std::function<const HashMap<int64_t, double>&(int64_t)>& PathIIScales4RegstDescId,
    const MemZoneRegstDescs& mz_regst_descs) const {
  double max_duration = CalcMaxRegstDescDuration(PathDurations4RegstDescId, mz_regst_descs);
  CHECK(!IsAnyZoneOutOfMemory(mz_regst_descs, PathDurations4RegstDescId, PathIIScales4RegstDescId,
                              max_duration));
  const double ii_search_threshold = 1;
  double r = max_duration;
  double l = base_ii;
  double mid = base_ii;
  while ((r - l) > ii_search_threshold) {
    mid = (l + r) / 2;
    if (IsAnyZoneOutOfMemory(mz_regst_descs, PathDurations4RegstDescId, PathIIScales4RegstDescId,
                             mid)) {
      l = mid;
    } else {
      r = mid;
    }
  }
  return r;
}

void Improver::MemoryLimitedAllocate(const ActGraph& graph,
                                     const std::function<void(int64_t, uint64_t)>& Handler) const {
  auto PathDurations4RegstDescId = MakeGetterPathDurations4RegstDescId(graph);
  auto PathIIScales4RegstDescId = MakeGetterPathIIScales4RegstDescId(graph);
  MemZoneRegstDescs mz_regst_descs;
  MakeMemZoneRegstDescs(graph.plan(), &mz_regst_descs);
  double ii = BinarySearchII(CalcBaseII(graph), PathDurations4RegstDescId, PathIIScales4RegstDescId,
                             mz_regst_descs);
  LOG(INFO) << "ii: " << ii;
  for (const auto& task_proto : graph.plan().task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      uint64_t regst_num =
          CalcRegstNum(pair.second, PathDurations4RegstDescId, ii, PathIIScales4RegstDescId);
      Handler(pair.second.regst_desc_id(), regst_num);
    }
  }
}

Plan Improver::Improve(const Plan& naive_plan, const std::string& act_event_filepath) {
  record_load_task_num_.assign(Global<JobDesc>::Get()->TotalMachineNum(), 0);
  for (const TaskProto& task_proto : naive_plan.task()) {
    if (task_proto.task_type() == TaskType::kRecordLoad) {
      record_load_task_num_.at(Global<IDMgr>::Get()->MachineId4ActorId(task_proto.task_id())) += 1;
    }
  }
  auto act_events = std::make_unique<std::list<ActEvent>>();
  ParseActEvents(act_event_filepath, act_events.get());
  ActGraph act_graph(naive_plan, std::move(act_events));
  PushAvgActTimeToProfiler(act_graph);
  Plan plan(naive_plan);
  MemoryLimitedAllocate(act_graph, MakeSetterSetPlanRegstNum(&plan));
  return plan;
}

}  // namespace oneflow
