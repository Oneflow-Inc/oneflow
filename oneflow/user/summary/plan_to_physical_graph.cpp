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
#include "oneflow/user/summary/plan_to_physical_graph.h"
#include "oneflow/core/summary/graph.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/framework/to_string.h"

namespace oneflow {

namespace summary {

void PlanToPhysicalGraphFile(const Plan& plan) {
  GraphDef physical_graph;
  physical_graph.set_version(3);  // "compute graph version number = 3"
  HashMap<int64_t, std::string> regst_desc_id2produce_op_name;
  HashMap<int64_t, std::string> task_id2op_name;
  HashSet<int64_t> ctrl_regst_desc_id_set;
  for (const TaskProto& task : plan.task()) {
    std::string op_name = "";
    for (const ExecNodeProto& exec_node : task.exec_sequence().exec_node()) {
      if (op_name != "") { op_name += " && "; }
      op_name += (exec_node.kernel_conf().op_attribute().op_conf().name());
    }
    if (op_name == "") { continue; }
    task_id2op_name.insert({task.task_id(), op_name});
    for (const auto& pair : task.produced_regst_desc()) {
      const RegstDescProto& regst = pair.second;
      int64_t regst_desc_id = regst.regst_desc_id();
      regst_desc_id2produce_op_name.insert({regst_desc_id, op_name});
      if (regst.regst_desc_type().has_ctrl_regst_desc()) {
        ctrl_regst_desc_id_set.insert(regst_desc_id);
      }
    }
  }

  for (const TaskProto& task : plan.task()) {
    if (task_id2op_name.find(task.task_id()) == task_id2op_name.end()) { continue; }
    NodeDef* node = physical_graph.add_node();
    node->set_name(task_id2op_name.at(task.task_id()));
    const OperatorConf& op_conf =
        task.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf();
    DeviceType device_type = Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(task.thrd_id());
    node->set_device(CHECK_JUST(DeviceTag4DeviceType(device_type)));
    if (op_conf.has_user_conf()) {
      const UserOpConf& user_op = op_conf.user_conf();
      node->set_op(user_op.op_type_name());
      node->mutable_attr()->insert(user_op.attr().begin(), user_op.attr().end());
    } else {
      // maybe need get op / attr by every different op_type_case
      node->set_op("system_op");
    }
    for (const auto& pair : task.consumed_regst_desc_id()) {
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        if (regst_desc_id2produce_op_name.find(regst_desc_id)
            != regst_desc_id2produce_op_name.end()) {
          std::string input_name = regst_desc_id2produce_op_name.at(regst_desc_id);
          if (ctrl_regst_desc_id_set.find(regst_desc_id) != ctrl_regst_desc_id_set.end()) {
            input_name = "^" + input_name;  // control edge
          }
          node->add_input(input_name);
        }
      }
    }
  }
  TeePersistentLogStream::Create("physical_graph")->Write(physical_graph);
}

}  // namespace summary

}  // namespace oneflow
