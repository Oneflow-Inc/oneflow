#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/graph/task_node.h"
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

std::function<const TaskProto&(int64_t)> PlanUtil::MakeGetterTaskProto4TaskId(const Plan& plan) {
  auto task_id2task_proto = std::make_shared<HashMap<int64_t, const TaskProto*>>();
  for (const TaskProto& task_proto : plan.task()) {
    task_id2task_proto->emplace(task_proto.task_id(), &task_proto);
  }
  return [task_id2task_proto](int64_t task_id) { return *task_id2task_proto->at(task_id); };
}

void PlanUtil::ToDotFile(const Plan& plan, const std::string& filepath) {
  size_t machine_num = Global<ResourceDesc>::Get()->TotalMachineNum();
  size_t gpu_device_num = Global<ResourceDesc>::Get()->GpuDeviceNum();
  std::vector<std::vector<std::vector<std::string>>> machine_id2device_id2node_list(machine_num);
  for (size_t i = 0; i < machine_num; ++i) {
    machine_id2device_id2node_list[i].resize(gpu_device_num);
  }
  std::vector<std::vector<std::string>> machine_id2host_node_list(machine_num);
  HashSet<int64_t> ctrl_regst_desc_ids;

  auto InsertNodeDefByTaskProto = [&](const TaskProto& task_proto, const std::string& node_def) {
    if (Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(task_proto.thrd_id()) == DeviceType::kGPU) {
      int64_t device_id = Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(task_proto.thrd_id());
      machine_id2device_id2node_list[task_proto.machine_id()][device_id].push_back(node_def);
    } else {
      machine_id2host_node_list[task_proto.machine_id()].push_back(node_def);
    }
  };

  auto GenNodeColorStr = [](const RegstDescTypeProto& type) {
    if (type.has_data_regst_desc()) {
      return ",color=\"black\",fillcolor=\"lightgray\"";
    } else if (type.has_ctrl_regst_desc()) {
      return ",color=\"gray70\",fillcolor=\"lightgray\"";
    } else {
      UNIMPLEMENTED();
    }
  };

  auto GenEdgeColorStr = [&](int64_t regst_desc_id) {
    if (ctrl_regst_desc_ids.find(regst_desc_id) != ctrl_regst_desc_ids.end()) {
      return ",fontcolor=\"gray70\",color=\"gray70\"";
    }
    return "";
  };

  auto log_stream = TeePersistentLogStream::Create(filepath);
  // task node
  for (const TaskProto& task_proto : plan.task()) {
    std::string node_def = "task" + std::to_string(task_proto.task_id()) + "[label=\"";
    for (const ExecNodeProto& exec_node : task_proto.exec_sequence().exec_node()) {
      node_def += (exec_node.kernel_conf().op_attribute().op_conf().name() + " ");
    }
    node_def +=
        ("\",tooltip=\"" + task_type2type_str.at(task_proto.task_type()) + "  "
         + std::to_string(task_proto.task_id()) + "-" + std::to_string(task_proto.machine_id())
         + ":" + std::to_string(task_proto.thrd_id()) + ":"
         + std::to_string(task_proto.parallel_ctx().parallel_id())
         + "\", shape=ellipse, style=\"rounded,filled\", "
         + "colorscheme=set312, color=" + std::to_string((task_proto.job_id() % 12) + 1) + "];\n");
    InsertNodeDefByTaskProto(task_proto, node_def);
  }
  // regst node
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      std::string node_def = "regst_desc" + std::to_string(pair.second.regst_desc_id())
                             + "[label=\"" + std::to_string(pair.second.regst_desc_id()) + "\""
                             + GenNodeColorStr(pair.second.regst_desc_type()) + ",tooltip=\""
                             + "regst_num = " + std::to_string(pair.second.register_num())
                             + "\",style=\"rounded,filled\",shape=\"box\"];\n";
      InsertNodeDefByTaskProto(task_proto, node_def);
      if (pair.second.regst_desc_type().has_ctrl_regst_desc()) {
        ctrl_regst_desc_ids.insert(pair.second.regst_desc_id());
      }
    }
  }
  log_stream << "digraph merged_plan_graph {\n splines=\"ortho\";\n";
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
      log_stream << "subgraph cluster_" << device_name << " { label = \"" << device_name << "\";\n";
      log_stream << "color=\"skyblue\";\n";
      log_stream << "fillcolor=\"azure\";\n";
      log_stream << "style=\"rounded,filled\";\n";
      for (const auto& device_node_def : machine_id2device_id2node_list[machine_id][device_id]) {
        log_stream << device_node_def;
      }
      log_stream << "}\n";
    }
    log_stream << "}\n";
  }

  // produce/consume edge
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      log_stream << "task" << std::to_string(task_proto.task_id()) << "->regst_desc"
                 << std::to_string(pair.second.regst_desc_id()) << "[xlabel=\"" << pair.first
                 << "\"" << GenEdgeColorStr(pair.second.regst_desc_id()) << "];\n";
    }
    for (const auto& pair : task_proto.consumed_regst_desc_id()) {
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        log_stream << "regst_desc" << std::to_string(regst_desc_id) << "->task"
                   << std::to_string(task_proto.task_id()) << "[xlabel=\"" << pair.first << "\""
                   << GenEdgeColorStr(regst_desc_id) << "];\n";
      }
    }
  }

  log_stream << "}\n";
}

}  // namespace oneflow
