#include "oneflow/core/compiler/compiler.h"
#include "oneflow/core/graph/data_comp_task_node.h"
#include "oneflow/core/graph/data_task_graph.h"
#include "oneflow/core/graph/model_diff_accumulate_task_graph.h"
#include "oneflow/core/graph/model_save_comp_task_node.h"
#include "oneflow/core/graph/model_save_task_graph.h"
#include "oneflow/core/graph/model_update_task_graph.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

namespace compiler {

void Compiler::ConstForEachChainNode(
    std::function<void(const ChainNode*)> func) {
  for (const auto& task_gph : ordered_task_gphs_) {
    task_gph->chain_gph()->ConstForEachNode(
        [&](const ChainNode* chain) { func(chain); });
  }
}

void Compiler::ConstForEachStageNode(
    std::function<void(const StageNode*)> func) {
  for (const auto& task_gph : ordered_task_gphs_) {
    task_gph->stage_gph()->ConstForEachNode(
        [&](const StageNode* stage) { func(stage); });
  }
}

void Compiler::ForEachTaskNode(std::function<void(TaskNode*)> func) {
  for (const auto& task_gph : ordered_task_gphs_) {
    task_gph->ForEachNode([&](TaskNode* task) { func(task); });
  }
}

// TODO: inference "register_num for each register_desc"
void Compiler::Compile(const JobConf& job_conf,
                       const std::string& plan_filepath) {
  JobDesc::Singleton()->InitFromJobConf(job_conf);
  IDMgr::Singleton()->InitFromResource(JobDesc::Singleton()->resource());
  BuildGraphs();
  InferShape4Regsts();
  EraseMeaningLessRegsts();
  GenPlanFile(plan_filepath);
}

void Compiler::Compile(const JobDescProto& job_desc, Plan* plan) {
  JobDesc::Singleton()->InitFromProto(job_desc);
  IDMgr::Singleton()->InitFromResource(JobDesc::Singleton()->resource());
  BuildGraphs();
  InferShape4Regsts();
  EraseMeaningLessRegsts();
  GenPlan(plan);
  // GenPlanFile(plan_filepath);
}

void Compiler::BuildGraphs() {
  ordered_task_gphs_.clear();
  // data graph
  LOG(INFO) << "Build DataTaskGraph";
  auto data_task_gph = new DataTaskGraph(
      "data", JobDesc::Singleton()->train_dlnet_conf(),
      JobDesc::Singleton()->placement(), JobDesc::Singleton()->is_train());
  ordered_task_gphs_.emplace_back(data_task_gph);
  // construct data_chain2sorted_fw_comp_tasks
  HashMap<const ChainNode*, std::vector<CompTaskNode*>>
      data_chain2sorted_fw_comp_tasks;
  data_task_gph->ForEachNode([&](TaskNode* node) {
    auto fw_node = dynamic_cast<CompTaskNode*>(node);
    if (fw_node == nullptr || fw_node->IsBpNode() || fw_node->IsLossNode()) {
      return;
    }
    data_chain2sorted_fw_comp_tasks[fw_node->chain_node()].push_back(fw_node);
  });
  for (auto& pair : data_chain2sorted_fw_comp_tasks) {
    SortByParallelId(&(pair.second));
  }
  // model graph
  for (const auto& pair : data_chain2sorted_fw_comp_tasks) {
    BuildModelGraphs(pair);
  }
  // all exec_graph 2 dot
  ForEachTaskNode(
      [](const TaskNode* node) { node->exec_gph().ToDotWithAutoFilePath(); });
}

void Compiler::BuildModelGraphs(
    const std::pair<const ChainNode*, std::vector<CompTaskNode*>>& pair) {
  if (pair.first->HasOpWithModelOrModelTmpBlob() == false) { return; }
  std::string chain_tag = pair.first->op_vec().front()->op_name();
  StringReplace(&chain_tag, '/', '_');
  ParallelPolicy policy = pair.first->parallel_desc()->policy();

  bool is_train = JobDesc::Singleton()->is_train();
  std::vector<CompTaskNode*> sorted_diff_acc_tasks;
  if (is_train) {
    LOG(INFO) << "Build MdDiffAccTaskGraph for " << chain_tag;
    auto diff_acc_gph = new MdDiffAccTaskGraph("md_diff_acc_" + chain_tag,
                                               pair.first, pair.second);
    ordered_task_gphs_.emplace_back(diff_acc_gph);

    ChainNode* diff_acc_chain = diff_acc_gph->chain_gph()->SoleSinkNode();
    sorted_diff_acc_tasks = diff_acc_gph->CompTasksInChain(diff_acc_chain);
    SortByParallelId(&sorted_diff_acc_tasks);
  }

  LOG(INFO) << "Build MdUpdtTaskGraph for " << chain_tag;
  std::vector<CompTaskNode*> updt_tasks;
  updt_tasks.reserve(pair.second.size());
  uint32_t random_seed = NewRandomSeed();
  for (size_t i = 0; i < pair.second.size(); ++i) {
    CompTaskNode* data_fw_task = pair.second[i];
    auto updt_gph = new MdUpdtTaskGraph(
        "md_updt_" + data_fw_task->node_id_str(), data_fw_task,
        is_train ? sorted_diff_acc_tasks[i] : nullptr, random_seed);
    ordered_task_gphs_.emplace_back(updt_gph);
    ChainNode* updt_chain = updt_gph->chain_gph()->SoleSinkNode();
    auto updt_tasks_in_chain = updt_gph->CompTasksInChain(updt_chain);
    CHECK_EQ(updt_tasks_in_chain.size(), 1);
    updt_tasks.push_back(updt_tasks_in_chain[0]);
  }

  if (is_train) {
    LOG(INFO) << "Build MdSaveTaskGraph for " << chain_tag;
    if (policy == kDataParallel) { updt_tasks = {updt_tasks.front()}; }
    for (CompTaskNode* update_task : updt_tasks) {
      auto save_gph = new MdSaveTaskGraph(
          "md_save_" + update_task->node_id_str(), update_task);
      ordered_task_gphs_.emplace_back(save_gph);
    }
  }
}

void Compiler::InferShape4Regsts() {
  for (auto& task_gph : ordered_task_gphs_) {
    LOG(INFO) << "InferShape for " << task_gph->name();
    task_gph->InferShapeOfBlobsInProducedRegsts();
  }
}

void Compiler::EraseMeaningLessRegsts() {
  ForEachTaskNode([](TaskNode* task_node) {
    task_node->EraseZeroSizeBlobInProducedRegsts();
    task_node->EraseProducedEmptyRegsts();
  });
}

void Compiler::GenPlan(Plan* plan) {
  HashMap<const ChainNode*, int64_t> chain2meaningless_task_cnt;
  ForEachTaskNode([&](const TaskNode* node) {
    auto comp_task_node = dynamic_cast<const DataCompTaskNode*>(node);
    if (comp_task_node && node->IsFwNode() && node->IsMeaningLess()) {
      chain2meaningless_task_cnt[node->chain_node()] += 1;
    }
  });
  auto MeaninglessTaskCnt4Chain = [&](const ChainNode* chain) -> int64_t {
    auto it = chain2meaningless_task_cnt.find(chain);
    if (it != chain2meaningless_task_cnt.end()) {
      return it->second;
    } else {
      return 0;
    }
  };
  ForEachTaskNode([&](const TaskNode* node) {
    if (node->IsMeaningLess()) {
      LOG(INFO) << "MeaningLess Task Id: " << node->task_id();
      return;
    }
    node->ToProto(plan->mutable_task()->Add(), MeaninglessTaskCnt4Chain);
  });

  OpMgr::Singleton()->AllOpToProto(plan->mutable_op());
  JobDesc::Singleton()->ToProto(plan->mutable_job_desc());
  ForEachTaskNode([&](const TaskNode* task_node) {
    task_node->exec_gph().ConstForEachNode([&](const ExecNode* exec_node) {
      const std::string& op_name = exec_node->op()->op_name();
      // op_name2device_type
      auto it = plan->mutable_op_name2device_type()->find(op_name);
      if (it == plan->mutable_op_name2device_type()->end()) {
        plan->mutable_op_name2device_type()->insert(
            {op_name, task_node->chain_node()->parallel_desc()->device_type()});
      } else {
        CHECK_EQ(it->second,
                 task_node->chain_node()->parallel_desc()->device_type());
      }
      // machine_id2op_name_set
      int64_t machine_id = task_node->stage_node()->machine_id();
      (*(plan->mutable_machine_id2op_name_set()))[machine_id].add_op_name(
          op_name);
      // TODO: unique
    });
  });
}

void Compiler::GenPlanFile(const std::string& plan_filepath) {
  Plan plan;
  GenPlan(&plan);
  PrintProtoToTextFile(plan, plan_filepath);
  Plan2DotFile(plan);
}

void Compiler::Plan2DotFile(const Plan& plan) {
  const std::string file_path = LogDir() + "/dot/plan.dot";
  PersistentOutStream out_stream(file_path);
  out_stream << "digraph {\n";
  HashSet<int64_t> regst_desc_ids;
  for (const TaskProto& task_proto : plan.task()) {
    out_stream << "task" << std::to_string(task_proto.id())
               << "[label=\"task_id:" << std::to_string(task_proto.id())
               << "\\nthrd_loc_id:"
               << std::to_string(task_proto.thrd_local_id())
               << "\\nparallel_id:" << std::to_string(task_proto.parallel_id())
               << "\", shape=ellipse];\n";
    for (const auto& pair : task_proto.produced_regst_desc()) {
      regst_desc_ids.insert(pair.second.regst_desc_id());
    }
  }
  for (const int64_t regst_task_id : regst_desc_ids) {
    out_stream << "regst_desc" << std::to_string(regst_task_id) << "[label=\""
               << std::to_string(regst_task_id) << "\", shape=box];\n";
  }
  for (const TaskProto& task_proto : plan.task()) {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      out_stream << "task" << std::to_string(task_proto.id()) << "->regst_desc"
                 << std::to_string(pair.second.regst_desc_id()) << "[label=\""
                 << pair.first << "\"];\n";
    }
    for (const auto& pair : task_proto.consumed_regst_desc_id()) {
      out_stream << "regst_desc" << std::to_string(pair.second) << "->task"
                 << std::to_string(task_proto.id()) << "[label=\"" << pair.first
                 << "\"];\n";
    }
  }
  out_stream << "}\n";
}

}  // namespace compiler

}  // namespace oneflow
