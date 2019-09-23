#include "oneflow/core/job/inter_job_mem_sharing_util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/plan_util.h"

namespace oneflow {

namespace {

std::vector<TaskProto*> SortSameOpNameTaskProtos(const std::string& op_name, Plan* plan) {
  std::vector<TaskProto*> task_protos;
  FOR_RANGE(int64_t, i, 0, plan->task_size()) {
    TaskProto* task = plan->mutable_task(i);
    if (task->exec_sequence().exec_node_size() == 1) {
      const KernelConf& kernel_conf = task->exec_sequence().exec_node(0).kernel_conf();
      if (op_name == kernel_conf.op_attribute().op_conf().name()) {
        CHECK(task->has_parallel_ctx());
        task_protos.emplace_back(task);
      }
    }
  }
  std::sort(task_protos.begin(), task_protos.end(), [](const TaskProto* lhs, const TaskProto* rhs) {
    return lhs->parallel_ctx().parallel_id() < rhs->parallel_ctx().parallel_id();
  });
  return task_protos;
}

HashMap<std::string, HashSet<int64_t>> GetInterfaceOpName2JobIds(const std::vector<Job>& jobs) {
  HashMap<std::string, HashSet<int64_t>> interface_op_name2job_ids;
  HashSet<std::string> unique_op_name_check;
  FOR_RANGE(int64_t, i, 0, jobs.size()) {
    const auto& job = jobs.at(i);
    for (const auto& op : job.net().op()) {
      if (IsInterfaceOpConf(op)) {
        CHECK(interface_op_name2job_ids[op.name()].emplace(i).second);
        unique_op_name_check.emplace(op.name());
      } else {
        // interface ops shouldn't share op_name with other ops
        CHECK(unique_op_name_check.find(op.name()) == unique_op_name_check.end());
      }
    }
  }
  return interface_op_name2job_ids;
}

std::vector<HashSet<int64_t>> InitJobId2MutualExclusionJobIds(const std::vector<Job>& jobs) {
  int64_t job_size = jobs.size();
  std::vector<HashSet<int64_t>> job_id2mutual_exclusion_ids(job_size);
  for (const auto& pair : GetInterfaceOpName2JobIds(jobs)) {
    for (int64_t first_id : pair.second) {
      for (int64_t second_id : pair.second) {
        if (first_id != second_id) { job_id2mutual_exclusion_ids[first_id].emplace(second_id); }
      }
    }
  }
  const InterJobReuseMemStrategy* strategy = Global<const InterJobReuseMemStrategy>::Get();
  if (strategy->has_custom_parallelism()) {
    auto* job_name2job_id = Global<JobName2JobId>::Get();
    for (const auto& group : strategy->custom_parallelism().nonparallel_group()) {
      for (const std::string& first_name : group.job_name()) {
        for (const std::string& second_name : group.job_name()) {
          if (first_name != second_name) {
            CHECK(job_name2job_id->find(first_name) != job_name2job_id->end());
            CHECK(job_name2job_id->find(second_name) != job_name2job_id->end());
            int64_t first_id = (*job_name2job_id)[first_name];
            int64_t second_id = (*job_name2job_id)[second_name];
            job_id2mutual_exclusion_ids[first_id].emplace(second_id);
          }
        }
      }
    }
  }
  return job_id2mutual_exclusion_ids;
}

std::vector<HashSet<int64_t>> GetMutualExclusionJobGroups(const std::vector<Job>& jobs) {
  int64_t job_size = jobs.size();
  std::vector<HashSet<int64_t>> job_groups;
  if (Global<const InterJobReuseMemStrategy>::Get()->has_reuse_mem_priority()) {
    job_groups.push_back(HashSet<int64_t>());
    FOR_RANGE(int64_t, i, 0, job_size) { job_groups.front().emplace(i); }
    return job_groups;
  }

  // default using parallelism_priority strategy
  std::vector<HashSet<int64_t>> job_id2mutual_exclusion_ids = InitJobId2MutualExclusionJobIds(jobs);
  std::vector<HashSet<int64_t>> job_id2enable_parallel_ids(job_size);
  FOR_RANGE(int64_t, i, 0, job_size) {
    FOR_RANGE(int64_t, j, 0, job_size) {
      if (job_id2mutual_exclusion_ids[i].find(j) == job_id2mutual_exclusion_ids[i].end()) {
        job_id2enable_parallel_ids[i].emplace(j);
      }
    }
  }
  int64_t mem_share_group_num = 0;
  std::vector<int64_t> job_id2mem_share_group_id(job_size, -1);
  FOR_RANGE(int64_t, this_job_id, 0, job_size) {
    HashSet<int64_t> mem_share_group_id_used;
    for (int64_t enable_parallel_job_id : job_id2enable_parallel_ids[this_job_id]) {
      int64_t group_id = job_id2mem_share_group_id[enable_parallel_job_id];
      if (group_id != -1) { mem_share_group_id_used.emplace(group_id); }
    }
    FOR_RANGE(int64_t, this_group_id, 0, mem_share_group_num) {
      if (mem_share_group_id_used.find(this_group_id) == mem_share_group_id_used.end()) {
        job_id2mem_share_group_id[this_job_id] = this_group_id;
        break;
      }
    }
    if (job_id2mem_share_group_id[this_job_id] == -1) {
      job_id2mem_share_group_id[this_job_id] = mem_share_group_num;
      ++mem_share_group_num;
      CHECK_LE(mem_share_group_num, job_size);
    }
  }

  job_groups.resize(mem_share_group_num);
  FOR_RANGE(int64_t, this_job_id, 0, job_size) {
    job_groups[job_id2mem_share_group_id[this_job_id]].emplace(this_job_id);
  }
  {
    HashSet<int64_t> job_id_unique_check;
    for (auto& job_group : job_groups) {
      for (int64_t job_id : job_group) { CHECK(job_id_unique_check.emplace(job_id).second); }
    }
  }
  return job_groups;
}

int64_t GenMemZoneUniqueId(int64_t machine_id, const MemoryCase& mem_case) {
  int64_t mem_zone_id = 1024;
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().has_cuda_pinned_mem()) {
      mem_zone_id = 1025 + mem_case.host_mem().cuda_pinned_mem().device_id();
    }
  } else {
    mem_zone_id = mem_case.device_cuda_mem().device_id();
  }
  return (machine_id << 32) | mem_zone_id;
}

// mem block id infomation
struct MbiInfo {
  MbiInfo() = default;
  MbiInfo(int64_t mem_size, int64_t mzuid, int64_t job_id) {
    this->mem_size = mem_size;
    this->mzuid = mzuid;
    this->job_id = job_id;
    regst_descs = std::vector<RegstDescProto*>();
  }

  int64_t mem_size;
  // mzuid = memory zone unique id
  int64_t mzuid;
  int64_t job_id;
  std::vector<RegstDescProto*> regst_descs;
};

std::vector<std::unique_ptr<MbiInfo>> InitMemBlockId2MbiInfo(std::vector<Plan>* sub_plans) {
  int64_t mem_block_id_max = Global<IDMgr>::Get()->NewMemBlockId();
  std::vector<std::unique_ptr<MbiInfo>> mem_block_id2mbi_info(mem_block_id_max);
  for (int64_t job_id = 0; job_id < sub_plans->size(); ++job_id) {
    Plan* sub_plan = &(sub_plans->at(job_id));
    for (int64_t i = 0; i < sub_plan->task_size(); ++i) {
      TaskProto* task = sub_plan->mutable_task(i);
      for (auto& pair : *(task->mutable_produced_regst_desc())) {
        RegstDescProto* regst_desc = &pair.second;
        // only handle memory reused
        if (!regst_desc->enable_reuse_mem()) { continue; }
        CHECK(regst_desc->regst_desc_type().has_data_regst_desc());
        int32_t mem_block_id = regst_desc->mem_block_id();
        CHECK_GE(mem_block_id, 0);
        int64_t mem_byte_size =
            RtRegstDesc(*regst_desc).TotalMainByteSize4AllRegst() + regst_desc->mem_block_offset();
        CHECK_GT(mem_byte_size, 0);
        int64_t mzuid = GenMemZoneUniqueId(task->machine_id(), regst_desc->mem_case());
        if (mem_block_id2mbi_info[mem_block_id] == nullptr) {
          mem_block_id2mbi_info[mem_block_id].reset(new MbiInfo(mem_byte_size, mzuid, job_id));
        } else {
          CHECK_EQ(mem_block_id2mbi_info[mem_block_id]->mzuid, mzuid);
          CHECK_EQ(mem_block_id2mbi_info[mem_block_id]->job_id, job_id);
        }
        // only handle kMemMax
        mem_block_id2mbi_info[mem_block_id]->mem_size =
            std::max(mem_block_id2mbi_info[mem_block_id]->mem_size, mem_byte_size);
        mem_block_id2mbi_info[mem_block_id]->regst_descs.push_back(regst_desc);
      }
    }
  }
  return mem_block_id2mbi_info;
}

}  // namespace

void InterJobMemSharingUtil::BindInterfaceMemBlockId(const std::vector<Job>& jobs,
                                                     std::vector<Plan>* sub_plans) {
  for (const auto& pair : GetInterfaceOpName2JobIds(jobs)) {
    std::vector<std::vector<TaskProto*>> same_op_name_sorted_task_protos;
    for (int64_t job_id : pair.second) {
      same_op_name_sorted_task_protos.push_back(
          SortSameOpNameTaskProtos(pair.first, &sub_plans->at(job_id)));
    }
    const auto& first_vec = same_op_name_sorted_task_protos.at(0);
    std::vector<MemoryCase> common_mem_case_vec(first_vec.size());
    std::transform(
        first_vec.cbegin(), first_vec.cend(), common_mem_case_vec.begin(),
        [](TaskProto* tp) { return PlanUtil::GetSoleProducedDataRegst(tp)->mem_case(); });
    for (const auto& task_protos : same_op_name_sorted_task_protos) {
      CHECK_EQ(task_protos.size(), first_vec.size());
      FOR_RANGE(int64_t, i, 0, first_vec.size()) {
        CHECK_EQ(task_protos.at(i)->machine_id(), first_vec.at(i)->machine_id());
        RegstDescProto* first_regst_desc = PlanUtil::GetSoleProducedDataRegst(first_vec.at(i));
        CHECK_EQ(first_regst_desc->mem_block_offset(), 0);
        RegstDescProto* regst_desc = PlanUtil::GetSoleProducedDataRegst(task_protos.at(i));
        MemoryCase common_mem_case;
        CHECK(MemoryCaseUtil::GetCommonMemoryCase(common_mem_case_vec.at(i), regst_desc->mem_case(),
                                                  &common_mem_case));
        common_mem_case_vec[i] = common_mem_case;
        CHECK(regst_desc->enable_reuse_mem() == false);
        CHECK_EQ(regst_desc->mem_block_offset(), 0);
        CHECK_NE(first_regst_desc->mem_block_id(), -1);
        regst_desc->set_mem_block_id(first_regst_desc->mem_block_id());

        int64_t separated_header_mem_size =
            RtRegstDesc(*first_regst_desc).TotalSeparatedHeaderByteSize4AllRegst();
        if (separated_header_mem_size > 0) {
          CHECK_EQ(separated_header_mem_size,
                   RtRegstDesc(*regst_desc).TotalSeparatedHeaderByteSize4AllRegst());
          if (first_regst_desc->separated_header_mem_block_id() == -1) {
            first_regst_desc->set_separated_header_mem_block_id(
                Global<IDMgr>::Get()->NewMemBlockId());
          }
          regst_desc->set_separated_header_mem_block_id(
              first_regst_desc->separated_header_mem_block_id());
        }
      }
    }
    for (const auto& task_protos : same_op_name_sorted_task_protos) {
      FOR_RANGE(int64_t, i, 0, task_protos.size()) {
        *PlanUtil::GetSoleProducedDataRegst(task_protos.at(i))->mutable_mem_case() =
            common_mem_case_vec.at(i);
      }
    }
  }
}

void InterJobMemSharingUtil::MergeMemBlockBetweenSubPlans(const std::vector<Job>& jobs,
                                                          std::vector<Plan>* sub_plans) {
  if (jobs.size() == 1) { return; }
  std::vector<std::unique_ptr<MbiInfo>> mem_block_id2mbi_info = InitMemBlockId2MbiInfo(sub_plans);
  std::vector<HashMap<int64_t, HashSet<int32_t>>> job_id2mzuid2mem_block_ids(jobs.size());
  for (int32_t mem_block_id = 0; mem_block_id < mem_block_id2mbi_info.size(); ++mem_block_id) {
    if (mem_block_id2mbi_info[mem_block_id] == nullptr) { continue; }
    int64_t job_id = mem_block_id2mbi_info[mem_block_id]->job_id;
    int64_t mzuid = mem_block_id2mbi_info[mem_block_id]->mzuid;
    job_id2mzuid2mem_block_ids[job_id][mzuid].emplace(mem_block_id);
  }
  std::vector<HashSet<int64_t>> job_groups = GetMutualExclusionJobGroups(jobs);

  auto MergeMemBlockIdR2L = [&](int32_t lhs, int32_t rhs) {
    CHECK_NE(lhs, rhs);
    MbiInfo* info_l = mem_block_id2mbi_info[lhs].get();
    MbiInfo* info_r = mem_block_id2mbi_info[rhs].get();
    CHECK_NE(info_l->job_id, info_r->job_id);
    CHECK_EQ(info_l->mzuid, info_r->mzuid);
    CHECK_GT(info_l->mem_size, 0);
    CHECK_GT(info_r->mem_size, 0);
    for (auto* regst_desc : info_r->regst_descs) { regst_desc->set_mem_block_id(lhs); }
    info_l->mem_size = std::max(info_l->mem_size, info_r->mem_size);
    mem_block_id2mbi_info[rhs].reset(nullptr);
  };

  auto GetMemBlockId7MemSizeList = [&](int64_t job_id, int64_t mzuid) {
    std::vector<std::pair<int64_t, int64_t>> ret;
    for (int64_t mem_block_id : job_id2mzuid2mem_block_ids[job_id][mzuid]) {
      ret.push_back({mem_block_id, mem_block_id2mbi_info[mem_block_id]->mem_size});
    }
    std::sort(ret.begin(), ret.end(),
              [&](const std::pair<int64_t, int64_t>& lhs, const std::pair<int64_t, int64_t>& rhs) {
                return lhs.second > rhs.second;
              });
    return ret;
  };

  auto InitMzuid2JobIdsInJobGroup = [&](const HashSet<int64_t>& job_group) {
    HashMap<int64_t, HashSet<int64_t>> mzuid2job_ids;
    for (int64_t job_id : job_group) {
      for (const auto& pair : job_id2mzuid2mem_block_ids[job_id]) {
        CHECK(mzuid2job_ids[pair.first].emplace(job_id).second);
      }
    }
    return mzuid2job_ids;
  };

  auto FindMaxMemBlockNumJobId = [&](const HashSet<int64_t>& job_group, int64_t mzuid) {
    int64_t max_mem_block_num_job_id = -1;
    int32_t max_mem_block_num = 0;
    for (int64_t job_id : job_group) {
      if (job_id2mzuid2mem_block_ids[job_id][mzuid].size() > max_mem_block_num) {
        max_mem_block_num_job_id = job_id;
        max_mem_block_num = job_id2mzuid2mem_block_ids[job_id][mzuid].size();
      }
    }
    CHECK_NE(max_mem_block_num_job_id, -1);
    return max_mem_block_num_job_id;
  };

  for (const auto& job_group : job_groups) {
    if (job_group.size() <= 1) { continue; }
    HashMap<int64_t, HashSet<int64_t>> mzuid2job_ids = InitMzuid2JobIdsInJobGroup(job_group);
    for (const auto& pair : mzuid2job_ids) {
      const HashSet<int64_t>& job_ids = pair.second;
      if (job_ids.size() <= 1) { continue; }
      int64_t mzuid = pair.first;
      int64_t merge_job_id = FindMaxMemBlockNumJobId(job_ids, mzuid);
      for (int64_t job_id : job_ids) {
        if (job_id == merge_job_id) { continue; }
        auto lhs_info = GetMemBlockId7MemSizeList(merge_job_id, mzuid);
        auto rhs_info = GetMemBlockId7MemSizeList(job_id, mzuid);
        CHECK_GE(lhs_info.size(), rhs_info.size());
        for (int64_t i = 0; i < rhs_info.size(); ++i) {
          MergeMemBlockIdR2L(lhs_info[i].first, rhs_info[i].first);
        }
      }
    }
  }
}

}  // namespace oneflow
