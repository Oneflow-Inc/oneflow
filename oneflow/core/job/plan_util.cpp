/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/common/constant.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"

namespace oneflow {

RegstDescProto* PlanUtil::GetSoleProducedDataRegst(TaskProto* task_proto) {
  RegstDescProto* ret = nullptr;
  for (auto& pair : *task_proto->mutable_produced_regst_desc()) {
    RegstDescProto* regst_desc = &pair.second;
    if (regst_desc->regst_desc_type().has_data_regst_desc()) {
      CHECK_ISNULL(ret);
      CHECK_EQ(regst_desc->regst_desc_type().data_regst_desc().lbi2blob_desc_size(), 1);
      ret = regst_desc;
    }
  }
  CHECK_NOTNULL(ret);
  return ret;
}

std::function<const TaskProto*(int64_t)> PlanUtil::MakeGetterTaskProto4TaskId(const Plan& plan) {
  auto task_id2task_proto = std::make_shared<HashMap<int64_t, const TaskProto*>>();
  for (const TaskProto& task_proto : plan.task()) {
    task_id2task_proto->emplace(task_proto.task_id(), &task_proto);
  }
  return [task_id2task_proto](int64_t task_id) { return task_id2task_proto->at(task_id); };
}

void PlanUtil::CleanUselessMemBlockAndCheckValid(Plan* plan) {
  HashMap<int64_t, ChunkProto> chunk_id2chunk;
  HashMap<int64_t, MemBlockProto> mem_block_id2mem_block;
  for (const auto& chunk : plan->block_chunk_list().chunk()) {
    CHECK(chunk_id2chunk.emplace(chunk.chunk_id(), chunk).second);
  }
  for (const auto& mem_block : plan->block_chunk_list().mem_block()) {
    CHECK(mem_block_id2mem_block.emplace(mem_block.mem_block_id(), mem_block).second);
  }
  plan->mutable_block_chunk_list()->clear_mem_block();

  HashMap<int64_t, HashSet<int64_t>> chunk_id2job_ids;
  HashMap<int64_t, HashSet<int64_t>> mem_block_id2job_ids;
  for (const auto& pair : chunk_id2chunk) {
    for (int64_t job_id : pair.second.job_id()) {
      CHECK(chunk_id2job_ids[pair.first].insert(job_id).second);
    }
  }
  for (const auto& pair : mem_block_id2mem_block) {
    for (int64_t job_id : pair.second.job_id()) {
      CHECK(mem_block_id2job_ids[pair.first].insert(job_id).second);
    }
  }

  HashSet<int64_t> valid_mem_block_ids;
  for (const TaskProto& task : plan->task()) {
    for (const auto& pair : task.produced_regst_desc()) {
      const RegstDescProto& regst = pair.second;
      RtRegstDesc rt_regst(regst);
      int64_t regst_size = rt_regst.TotalMainByteSize4AllRegst();
      CHECK(mem_block_id2mem_block.find(regst.mem_block_id()) != mem_block_id2mem_block.end());
      const MemBlockProto& mem_block = mem_block_id2mem_block.at(regst.mem_block_id());
      CHECK_GE(mem_block.mem_size(), regst.mem_block_offset() + regst_size);
      CHECK_EQ(task.machine_id(), mem_block.machine_id());
      CHECK_EQ(mem_block.enable_reuse_mem(), regst.enable_reuse_mem());
      CHECK(mem_block.mem_case() == regst.mem_case());
      const auto& job_ids = mem_block_id2job_ids[regst.mem_block_id()];
      CHECK(job_ids.find(task.job_id()) != job_ids.end());
      valid_mem_block_ids.insert(regst.mem_block_id());

      // separated_header
      int64_t separated_header_mem_size = rt_regst.TotalSeparatedHeaderByteSize4AllRegst();
      if (separated_header_mem_size > 0) {
        int64_t header_block_id = regst.separated_header_mem_block_id();
        CHECK_NE(header_block_id, -1);
        CHECK(mem_block_id2mem_block.find(header_block_id) != mem_block_id2mem_block.end());
        const MemBlockProto& header_mem_block = mem_block_id2mem_block.at(header_block_id);
        CHECK_EQ(header_mem_block.mem_size(), separated_header_mem_size);
        CHECK_EQ(task.machine_id(), header_mem_block.machine_id());
        CHECK(header_mem_block.mem_case()
              == MemoryCaseUtil::GetHostPinnedMemoryCaseForRegstSeparatedHeader(regst.mem_case()));
        CHECK(header_mem_block.enable_reuse_mem() == false);
        const auto& header_block_job_ids = mem_block_id2job_ids[header_block_id];
        CHECK(header_block_job_ids.find(task.job_id()) != header_block_job_ids.end());
        valid_mem_block_ids.insert(regst.separated_header_mem_block_id());
      }
    }
  }

  HashSet<int64_t> useless_mem_block_ids;
  HashSet<int64_t> valid_chunk_ids;
  for (const auto& pair : mem_block_id2mem_block) {
    if (valid_mem_block_ids.find(pair.first) == valid_mem_block_ids.end()) {
      CHECK(useless_mem_block_ids.insert(pair.first).second);
      continue;
    }
    const MemBlockProto& mem_block = pair.second;
    if (mem_block.has_chunk_id()) {
      CHECK(mem_block.has_chunk_offset());
      CHECK(mem_block.enable_reuse_mem());
      CHECK(chunk_id2chunk.find(mem_block.chunk_id()) != chunk_id2chunk.end());
      const ChunkProto& chunk = chunk_id2chunk.at(mem_block.chunk_id());
      CHECK_GE(chunk.mem_size(), mem_block.chunk_offset() + mem_block.mem_size());
      CHECK_EQ(mem_block.job_id_size(), 1);
      CHECK_GE(chunk.job_id_size(), 1);
      const HashSet<int64_t>& chunk_job_ids = chunk_id2job_ids.at(chunk.chunk_id());
      CHECK(chunk_job_ids.find(mem_block.job_id(0)) != chunk_job_ids.end());
      valid_chunk_ids.insert(mem_block.chunk_id());
    }
  }
  CHECK_EQ(valid_chunk_ids.size(), chunk_id2chunk.size());

  for (int64_t useless_block_id : useless_mem_block_ids) {
    mem_block_id2mem_block.erase(useless_block_id);
  }

  for (const auto& pair : mem_block_id2mem_block) {
    *(plan->mutable_block_chunk_list()->add_mem_block()) = pair.second;
  }
}

void PlanUtil::ToDotFile(const Plan& plan, const std::string& filepath) {
  size_t machine_num = Global<ResourceDesc, ForSession>::Get()->TotalMachineNum();
  size_t gpu_device_num = Global<ResourceDesc, ForSession>::Get()->GpuDeviceNum();
  std::vector<std::vector<std::vector<std::string>>> machine_id2device_id2node_list(machine_num);
  for (size_t i = 0; i < machine_num; ++i) {
    machine_id2device_id2node_list[i].resize(gpu_device_num);
  }
  std::vector<std::vector<std::string>> machine_id2host_node_list(machine_num);
  std::vector<std::string> main_node_list;
  std::vector<std::string> copy_comm_net_node_list;
  HashSet<int64_t> ctrl_regst_desc_ids;
  HashMap<int64_t, HashMap<int64_t, std::string>> task_id2consumer_regst_id2name;
  HashMap<int64_t, std::string> task_id2op_name;

  auto InsertNodeDefByTaskProto = [&](const TaskProto& task_proto, const std::string& node_def,
                                      const std::string& pass_tag) {
    if (task_proto.task_type() == TaskType::kCopyCommNet) {
      copy_comm_net_node_list.push_back(node_def);
      return;
    }
    if (pass_tag == kNoPassTag) {
      if (Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(task_proto.thrd_id()) == DeviceType::kGPU) {
        int64_t device_id = Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(task_proto.thrd_id());
        machine_id2device_id2node_list[task_proto.machine_id()][device_id].push_back(node_def);
      } else {
        machine_id2host_node_list[task_proto.machine_id()].push_back(node_def);
      }
    } else if (pass_tag == kMainOp) {
      main_node_list.push_back(node_def);
    } else {
      UNIMPLEMENTED();
    }
  };

  auto GenEdgeColorStr = [](const RegstDescTypeProto& type) {
    if (type.has_ctrl_regst_desc()) { return "fontcolor=\"gray65\",color=\"gray65\""; }
    return "fontcolor=\"gray15\",color=\"gray15\"";
  };

  auto IsEsac2ReentrantLockEdge = [](const std::string& src_name, const std::string& dst_name) {
    if (src_name.find("Esac") != std::string::npos
        && dst_name.find("ReentrantLock") != std::string::npos) {
      return true;
    }
    return false;
  };

  auto IsEsacNode = [](const std::string& name) {
    if (name.find("Esac") != std::string::npos) { return true; }
    return false;
  };

  auto log_stream = TeePersistentLogStream::Create(filepath);
  // task node
  for (const TaskProto& task_proto : plan.task()) {
    std::string node_def = "task" + std::to_string(task_proto.task_id()) + "[label=\"{{";
    node_def += std::to_string(task_proto.task_id()) + ":" + std::to_string(task_proto.machine_id())
                + "\\n";
    std::string op_name = "";
    std::string pass_tag = kNoPassTag;
    for (const ExecNodeProto& exec_node : task_proto.exec_sequence().exec_node()) {
      const auto& op_conf = exec_node.kernel_conf().op_attribute().op_conf();
      op_name += op_conf.name();
      if (op_conf.has_pass_tag()) { pass_tag = op_conf.pass_tag(); }
    }
    task_id2op_name[task_proto.task_id()] = op_name;
    node_def += op_name;
    size_t index = 0;
    for (const auto& pair : task_proto.produced_regst_desc()) {
      std::string regst_id = std::to_string(pair.second.regst_desc_id());
      if (index % 2 == 0) {
        node_def += "}|{";
      } else {
        node_def += "|";
      }
      // node_def += "<regst_desc_" + regst_id + ">";
      node_def += (pair.first + ":" + regst_id + ":" + std::to_string(pair.second.register_num()));
      ++index;
    }
    node_def += "}}";
    node_def +=
        ("\",tooltip=\"" + TaskType_Name(task_proto.task_type()) + "  "
         + std::to_string(task_proto.task_id()) + "-" + std::to_string(task_proto.machine_id())
         + ":" + std::to_string(task_proto.thrd_id()) + ":"
         + std::to_string(task_proto.parallel_ctx().parallel_id())
         + "\", shape=record, style=\"rounded,filled\""
         + ",colorscheme=set312, fillcolor=" + std::to_string((task_proto.job_id() % 12) + 1));
    if (IsEsacNode(op_name)) { node_def += ",width=5,height=1.5"; }
    node_def += "];\n";
    InsertNodeDefByTaskProto(task_proto, node_def, pass_tag);
    for (const auto& pair : task_proto.consumed_regst_desc_id()) {
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        task_id2consumer_regst_id2name[task_proto.task_id()][regst_desc_id] = pair.first;
      }
    }
  }

  log_stream << "digraph merged_plan_graph {\n";
  log_stream << "#splines=\"ortho\";\n";
  log_stream << "#rankdir=TB;\n";
  log_stream << "#nodesep=1.3;\n";
  log_stream << "#ranksep=1.3;\n";
  log_stream << "node[color=\"gray\"];\n";
  // main_node and copy_comm_net_node graph
  for (const std::string& main_node : main_node_list) { log_stream << main_node; }
  for (const std::string& copy_comm_net_node : copy_comm_net_node_list) {
    log_stream << copy_comm_net_node;
  }
  // sub graph
  for (size_t machine_id = 0; machine_id < machine_num; ++machine_id) {
    std::string machine_name = "machine_" + std::to_string(machine_id);
    log_stream << "subgraph cluster_" << machine_name << " { label = \"" << machine_name << "\";\n";
    log_stream << "style=\"rounded\";\n";
    for (const std::string& host_node_def : machine_id2host_node_list[machine_id]) {
      log_stream << host_node_def;
    }
    for (size_t device_id = 0; device_id < gpu_device_num; ++device_id) {
      std::string device_name = machine_name + "_device_" + std::to_string(device_id);
      log_stream << "#subgraph cluster_" << device_name << " { label = \"" << device_name
                 << "\";\n";
      log_stream << "#color=\"skyblue\";\n";
      log_stream << "#fillcolor=\"azure\";\n";
      log_stream << "#style=\"rounded,filled\";\n";
      for (const auto& device_node_def : machine_id2device_id2node_list[machine_id][device_id]) {
        log_stream << device_node_def;
      }
      log_stream << "#}\n";
    }
    log_stream << "}\n";
  }

  // produce/consume edge
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      const RegstDescProto& regst = pair.second;
      std::string src_node = "task" + std::to_string(task_proto.task_id());
      // src_node += ":regst_desc_" + std::to_string(regst.regst_desc_id());
      for (int64_t consumer_task_id : regst.consumer_task_id()) {
        std::string dst_node = "task" + std::to_string(consumer_task_id);
        // dst_node +=  ":task_node_" + std::to_string(consumer_task_id);
        std::string consumer_regst_name =
            task_id2consumer_regst_id2name[consumer_task_id][regst.regst_desc_id()];
        std::string consumer_op_name = task_id2op_name[consumer_task_id];
        std::string producer_regst_name = pair.first;
        std::string producer_op_name = task_id2op_name[task_proto.task_id()];
        std::string tooltip = producer_op_name + " : " + producer_regst_name + " -> "
                              + consumer_op_name + " : " + consumer_regst_name;
        if (IsEsac2ReentrantLockEdge(producer_op_name, consumer_op_name)) {
          log_stream << dst_node << "->" << src_node
                     << "[arrowhead=\"invempty\",fontcolor=\"red\",color=\"red\",taillabel=\""
                     << consumer_regst_name << "\",tailtooltip=\"" << tooltip;
        } else {
          log_stream << src_node << "->" << dst_node << "["
                     << GenEdgeColorStr(regst.regst_desc_type()) << ",headlabel=\""
                     << consumer_regst_name << "\",headtooltip=\"" << tooltip;
        }
        log_stream << "\",tooltip=\"" << tooltip << "\",arrowsize=0.5,labeldistance=1.5,penwidth=2"
                   << "];\n";
      }
    }
  }
  log_stream << "}\n";
}

std::function<RegstDescProto*(int64_t)> PlanUtil::MakeMutRegstDesc4Id(Plan* plan) {
  auto regst_desc_id2regst_desc = std::make_shared<HashMap<int64_t, RegstDescProto*>>();
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      int64_t regst_desc_id = pair.second.regst_desc_id();
      regst_desc_id2regst_desc->insert({regst_desc_id, &pair.second});
    }
  }
  return [regst_desc_id2regst_desc](int64_t regst_desc_id) -> RegstDescProto* {
    return regst_desc_id2regst_desc->at(regst_desc_id);
  };
}

void PlanUtil::SetForceInplaceMemBlock(Plan* plan) {
  auto RegstDesc4Id = MakeMutRegstDesc4Id(plan);
  for (int i = 0; i < plan->task_size(); i++) {
    TaskProto* task = plan->mutable_task(i);
    for (auto& pair : *task->mutable_produced_regst_desc()) {
      RegstDescProto* regst_desc = &pair.second;
      if (regst_desc->has_force_inplace_consumed_regst_desc_id()) {
        int64_t force_id = regst_desc->force_inplace_consumed_regst_desc_id();
        const RegstDescProto* in_regst_desc = RegstDesc4Id(force_id);
        CHECK(!in_regst_desc->enable_reuse_mem());
        CHECK(!regst_desc->enable_reuse_mem());
        CHECK_NE(in_regst_desc->mem_block_id(), -1);
        CHECK_EQ(in_regst_desc->mem_block_offset(), 0);
        CHECK_EQ(regst_desc->mem_block_offset(), 0);
        CHECK_EQ(in_regst_desc->register_num(), regst_desc->register_num());
        regst_desc->set_mem_block_id(in_regst_desc->mem_block_id());
        regst_desc->set_inplace_consumed_regst_desc_id(force_id);
      }
    }
  }
}

}  // namespace oneflow
