#include "oneflow/core/job/inter_job_mem_sharing_util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/memory/memory_case.pb.h"
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
        task_protos.emplace_back(task);
      }
    }
  }
  std::sort(task_protos.begin(), task_protos.end(), [](const TaskProto* lhs, const TaskProto* rhs) {
    return lhs->machine_id() < rhs->machine_id()
           || (lhs->machine_id() == rhs->machine_id() && lhs->thrd_id() < rhs->thrd_id());
  });
  return task_protos;
}

HashSet<std::string> GetArgOpNames(const std::vector<Job>& jobs) {
  HashSet<std::string> arg_op_names;
  for (const Job& job : jobs) {
    for (const auto& arg_op_name : job.job_conf().arg_op_name()) {
      arg_op_names.insert(arg_op_name);
    }
    for (const OperatorConf& op_conf : job.net().op()) {
      if (op_conf.has_variable_conf()) { arg_op_names.insert(op_conf.name()); }
    }
  }
  return arg_op_names;
}

HashMap<std::string, HashSet<int32_t>> GetInterfaceOpName2JobIds(const std::vector<Job>& jobs) {
  HashSet<std::string> arg_op_names = GetArgOpNames(jobs);
  HashMap<std::string, HashSet<int32_t>> interface_op_name2job_ids;
  HashSet<std::string> unique_op_name_check;
  FOR_RANGE(int32_t, i, 0, jobs.size()) {
    const auto& job = jobs.at(i);
    for (const auto& op : job.net().op()) {
      if (IsInterfaceOpConf(op)) {
        CHECK(arg_op_names.find(op.name()) != arg_op_names.end());
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

std::vector<HashSet<int32_t>> GetMutualExclusionJobGroups(const std::vector<Job>& jobs) {
  int32_t job_size = jobs.size();
  std::vector<HashSet<int32_t>> job_groups;
  std::vector<HashSet<int32_t>> job_id2mutual_exclusion_ids(job_size);
  for (const auto& pair : GetInterfaceOpName2JobIds(jobs)) {
    for (int32_t first_id : pair.second) {
      for (int32_t second_id : pair.second) {
        if (first_id != second_id) { job_id2mutual_exclusion_ids[first_id].emplace(second_id); }
      }
    }
  }

  const JobMemSharingStrategy* strategy = Global<const JobMemSharingStrategy>::Get();
  if (strategy->has_mem_sharing_priority()) {
    job_groups.push_back(HashSet<int32_t>());
    FOR_RANGE(int32_t, i, 0, job_size) { job_groups.front().emplace(i); }
    return job_groups;
  } else if (strategy->has_parallelism_priority()) {
    // do nothing
  } else if (strategy->has_custom_parallelism()) {
    auto* job_name2job_id = Global<JobName2JobId>::Get();
    for (const auto& group : strategy->custom_parallelism().nonparallel_group()) {
      for (const std::string& first_name : group.job_name()) {
        for (const std::string& second_name : group.job_name()) {
          if (first_name != second_name) {
            CHECK(job_name2job_id->find(first_name) != job_name2job_id->end());
            CHECK(job_name2job_id->find(second_name) != job_name2job_id->end());
            int32_t first_id = (*job_name2job_id)[first_name];
            int32_t second_id = (*job_name2job_id)[second_name];
            job_id2mutual_exclusion_ids[first_id].emplace(second_id);
          }
        }
      }
    }
  } else {
    // do nothing; default using parallelism_priority strategy
  }

  std::vector<HashSet<int32_t>> job_id2enable_parallel_ids;
  job_id2enable_parallel_ids.resize(job_size);
  FOR_RANGE(int32_t, i, 0, job_size) {
    FOR_RANGE(int32_t, j, 0, job_size) {
      if (job_id2mutual_exclusion_ids[i].find(j) == job_id2mutual_exclusion_ids[i].end()) {
        job_id2enable_parallel_ids[i].emplace(j);
      }
    }
  }

  int32_t mem_share_group_num = 0;
  std::vector<int32_t> job_id2mem_share_group_id(job_size, -1);
  FOR_RANGE(int32_t, this_job_id, 0, job_size) {
    HashSet<int32_t> mem_share_group_id_used;
    for (int32_t enable_parallel_job_id : job_id2enable_parallel_ids[this_job_id]) {
      int32_t group_id = job_id2mem_share_group_id[enable_parallel_job_id];
      if (group_id != -1) { mem_share_group_id_used.emplace(group_id); }
    }
    FOR_RANGE(int32_t, this_group_id, 0, mem_share_group_num) {
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
  FOR_RANGE(int32_t, this_job_id, 0, job_size) {
    job_groups[job_id2mem_share_group_id[this_job_id]].emplace(this_job_id);
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

}  // namespace

void InterJobMemSharingUtil::BindInterfaceMemBlockId(const std::vector<Job>& jobs,
                                                     std::vector<Plan>* sub_plans) {
  for (const auto& pair : GetInterfaceOpName2JobIds(jobs)) {
    std::vector<std::vector<TaskProto*>> same_op_name_sorted_task_protos;
    for (int32_t job_id : pair.second) {
      same_op_name_sorted_task_protos.push_back(
          SortSameOpNameTaskProtos(pair.first, &sub_plans->at(job_id)));
    }
    const auto& first_vec = same_op_name_sorted_task_protos.at(0);
    for (const auto& task_protos : same_op_name_sorted_task_protos) {
      CHECK_EQ(task_protos.size(), first_vec.size());
      FOR_RANGE(int32_t, i, 0, first_vec.size()) {
        CHECK_EQ(task_protos.at(i)->machine_id(), first_vec.at(i)->machine_id());
        CHECK_EQ(task_protos.at(i)->thrd_id(), first_vec.at(i)->thrd_id());
        RegstDescProto* first_regst_desc = PlanUtil::GetSoleProducedDataRegst(first_vec.at(i));
        CHECK_EQ(first_regst_desc->mem_shared_offset(), 0);
        RegstDescProto* regst_desc = PlanUtil::GetSoleProducedDataRegst(task_protos.at(i));
        CHECK_EQ(regst_desc->mem_shared_offset(), 0);
        CHECK_NE(first_regst_desc->mem_shared_id(), -1);
        regst_desc->set_mem_shared_id(first_regst_desc->mem_shared_id());

        int64_t separated_header_mem_size =
            RtRegstDesc(*first_regst_desc).TotalSeparatedHeaderByteSize4AllRegst();
        if (separated_header_mem_size > 0) {
          CHECK_EQ(separated_header_mem_size,
                   RtRegstDesc(*regst_desc).TotalSeparatedHeaderByteSize4AllRegst());
          if (first_regst_desc->separated_header_mem_block_id() == -1) {
            first_regst_desc->set_separated_header_mem_block_id(
                Global<IDMgr>::Get()->NewMemSharedId());
          }
          regst_desc->set_separated_header_mem_block_id(
              first_regst_desc->separated_header_mem_block_id());
        }
      }
    }
  }
}

void InterJobMemSharingUtil::MergeMemBlockBetweenSubPlans(const std::vector<Job>& jobs,
                                                          std::vector<Plan>* sub_plans) {
  int32_t job_size = jobs.size();
  if (job_size == 1) { return; }
  int64_t mem_block_id_max = Global<IDMgr>::Get()->NewMemSharedId();
  std::vector<std::vector<RegstDescProto*>> mem_block_id2regst_descs(mem_block_id_max);
  std::vector<int64_t> mem_block_id2size(mem_block_id_max, -1);
  // mzui = memory zone unique id
  std::vector<int64_t> mem_block_id2mzui(mem_block_id_max, -1);
  std::vector<int32_t> mem_block_id2job_id(mem_block_id_max, -1);
  std::vector<HashMap<int64_t, HashSet<int32_t>>> job_id2mzui2mem_block_ids(job_size);

  for (int32_t job_id = 0; job_id < job_size; ++job_id) {
    Plan* sub_plan = &(sub_plans->at(job_id));
    for (int32_t i = 0; i < sub_plan->task_size(); ++i) {
      TaskProto* task = sub_plan->mutable_task(i);
      for (auto& pair : *(task->mutable_produced_regst_desc())) {
        RegstDescProto* regst_desc = &pair.second;
        // only handle memory reused
        if (!regst_desc->enable_mem_sharing()) { continue; }
        CHECK(regst_desc->regst_desc_type().has_data_regst_desc());
        int32_t mem_block_id = regst_desc->mem_shared_id();
        CHECK_GE(mem_block_id, 0);
        int64_t mem_byte_size =
            RtRegstDesc(*regst_desc).TotalMainByteSize4AllRegst() + regst_desc->mem_shared_offset();
        CHECK_GT(mem_byte_size, 0);
        int64_t mzui = GenMemZoneUniqueId(task->machine_id(), regst_desc->mem_case());
        if (mem_block_id2size[mem_block_id] == -1) {
          mem_block_id2mzui[mem_block_id] = mzui;
          mem_block_id2job_id[mem_block_id] = job_id;
        } else {
          CHECK_EQ(mem_block_id2mzui[mem_block_id], mzui);
          CHECK_EQ(mem_block_id2job_id[mem_block_id], job_id);
        }
        // only handle kMemMax
        mem_block_id2size[mem_block_id] = std::max(mem_block_id2size[mem_block_id], mem_byte_size);
        mem_block_id2regst_descs[mem_block_id].push_back(regst_desc);
        job_id2mzui2mem_block_ids[job_id][mzui].emplace(mem_block_id);
      }
    }
  }

  std::vector<HashSet<int32_t>> job_groups = GetMutualExclusionJobGroups(jobs);
  HashSet<int32_t> job_id_unique;
  for (auto& job_group : job_groups) {
    for (int32_t job_id : job_group) { CHECK(job_id_unique.emplace(job_id).second); }
  }
  // merge
  auto MergeMemBlockIdR2L = [&](int32_t lhs, int32_t rhs) {
    CHECK_NE(lhs, rhs);
    CHECK_NE(mem_block_id2job_id[lhs], mem_block_id2job_id[rhs]);
    CHECK_EQ(mem_block_id2mzui[lhs], mem_block_id2mzui[rhs]);
    CHECK_GT(mem_block_id2size[lhs], 0);
    CHECK_GT(mem_block_id2size[rhs], 0);
    for (auto* regst_desc : mem_block_id2regst_descs[rhs]) { regst_desc->set_mem_shared_id(lhs); }
    mem_block_id2size[lhs] = std::max(mem_block_id2size[lhs], mem_block_id2size[rhs]);
    mem_block_id2size[rhs] = -1;
    mem_block_id2mzui[rhs] = -1;
    mem_block_id2job_id[rhs] = -1;
  };

  auto GetMemBlockIdsInfo = [&](int32_t job_id, int64_t mzui) {
    std::vector<std::pair<int64_t, int64_t>> ret;
    for (int64_t mem_block_id : job_id2mzui2mem_block_ids[job_id][mzui]) {
      ret.push_back({mem_block_id, mem_block_id2size[mem_block_id]});
    }
    std::sort(ret.begin(), ret.end(),
              [&](const std::pair<int64_t, int64_t>& lhs, const std::pair<int64_t, int64_t>& rhs) {
                return lhs.second > rhs.second;
              });
    return ret;
  };

  for (auto& job_group : job_groups) {
    if (job_group.size() <= 1) { continue; }
    int32_t max_mzui_num_job_id = *(job_group.begin());
    int32_t max_mzui_num = job_id2mzui2mem_block_ids[max_mzui_num_job_id].size();
    for (int32_t job_id : job_group) {
      if (job_id2mzui2mem_block_ids[job_id].size() > max_mzui_num) {
        max_mzui_num_job_id = job_id;
        max_mzui_num = job_id2mzui2mem_block_ids[job_id].size();
      }
    }
    // separate merge mem_block_id by different memory zone unique id,
    // for merge as much memory as possible
    for (const auto& pair : job_id2mzui2mem_block_ids[max_mzui_num_job_id]) {
      int64_t mzui = pair.first;
      int32_t max_mem_block_num = pair.second.size();
      int32_t merge_job_id = max_mzui_num_job_id;
      for (int32_t job_id : job_group) {
        auto& mzui2mem_block_ids = job_id2mzui2mem_block_ids[job_id];
        if (mzui2mem_block_ids.find(mzui) != mzui2mem_block_ids.end()) {
          int32_t mem_block_num = mzui2mem_block_ids[mzui].size();
          if (mem_block_num > max_mem_block_num) {
            max_mem_block_num = mem_block_num;
            merge_job_id = job_id;
          }
        }
      }

      for (int32_t job_id : job_group) {
        if (job_id == merge_job_id) { continue; }
        auto& mzui2mem_block_ids = job_id2mzui2mem_block_ids[job_id];
        if (mzui2mem_block_ids.find(mzui) == mzui2mem_block_ids.end()) { continue; }
        auto lhs_info = GetMemBlockIdsInfo(merge_job_id, mzui);
        auto rhs_info = GetMemBlockIdsInfo(job_id, mzui);
        CHECK_GE(lhs_info.size(), rhs_info.size());
        for (int64_t i = 0; i < rhs_info.size(); ++i) {
          MergeMemBlockIdR2L(lhs_info[i].first, rhs_info[i].first);
        }
      }
    }
  }
}

}  // namespace oneflow
