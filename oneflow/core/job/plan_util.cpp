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
  auto log_stream = TeePersistentLogStream::Create(filepath);
  log_stream << "digraph {\n";
  HashMap<int64_t, std::string> regst_desc_id2node_shape;
  auto GenNodeShapeStr = [](const RegstDescTypeProto& type) {
    if (type.has_data_regst_desc()) {
      return "shape=box";
    } else if (type.has_ctrl_regst_desc()) {
      return "shape=triangle";
    } else {
      UNIMPLEMENTED();
    }
  };

  for (const TaskProto& task_proto : plan.task()) {
    log_stream << "task" << std::to_string(task_proto.task_id()) << "[label=\"";
    for (const ExecNodeProto& exec_node : task_proto.exec_sequence().exec_node()) {
      log_stream << exec_node.kernel_conf().op_attribute().op_conf().name() << " ";
    }
    log_stream << "\",tooltip=\"" << task_type2type_str.at(task_proto.task_type()) << "  "
               << std::to_string(task_proto.task_id()) << "-"
               << std::to_string(task_proto.machine_id()) << ":"
               << std::to_string(task_proto.thrd_id()) << ":"
               << std::to_string(task_proto.parallel_ctx().parallel_id())
               << "\", shape=ellipse, style=\"rounded,filled\", "
                  "colorscheme=set312, color="
               << task_type2color.at(task_proto.task_type()) << "];\n";
    for (const auto& pair : task_proto.produced_regst_desc()) {
      regst_desc_id2node_shape.emplace(pair.second.regst_desc_id(),
                                       GenNodeShapeStr(pair.second.regst_desc_type()));
    }
  }
  for (const auto& pair : regst_desc_id2node_shape) {
    log_stream << "regst_desc" << std::to_string(pair.first) << "[label=\""
               << std::to_string(pair.first) << "\", " << pair.second << " ];\n";
  }
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      log_stream << "task" << std::to_string(task_proto.task_id()) << "->regst_desc"
                 << std::to_string(pair.second.regst_desc_id()) << "[label=\"" << pair.first
                 << "\"];\n";
    }
    for (const auto& pair : task_proto.consumed_regst_desc_id()) {
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        log_stream << "regst_desc" << std::to_string(regst_desc_id) << "->task"
                   << std::to_string(task_proto.task_id()) << "[label=\"" << pair.first << "\"];\n";
      }
    }
  }
  log_stream << "}\n";
}

}  // namespace oneflow
