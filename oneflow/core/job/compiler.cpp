#include "gflags/gflags.h"
#include "glog/logging.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/graph/model_save_comp_task_node.h"
#include "oneflow/core/graph/model_save_task_graph.h"
#include "oneflow/core/graph/model_update_task_graph.h"
#include "oneflow/core/graph/data_task_graph.h"
#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class Compiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Compiler);
  ~Compiler() = default;

  static Compiler& Singleton() {
    static Compiler obj;
    return obj;
  }

  void Compile(const JobConf& job_conf, const std::string& plan_filepath);

 private:
  Compiler() = default;
  void ConstForEachChainNode(std::function<void(const ChainNode*)> func);
  void ConstForEachStageNode(std::function<void(const StageNode*)> func);
  void ForEachTaskNode(std::function<void(TaskNode*)> func);
  
  void BuildGraphs();
  void BuildModelGraphs(const std::pair<const ChainNode*, std::vector<CompTaskNode*>>&);
  void InferShape4Regsts();
  void EraseMeaningLessRegsts();
  void GenPlanFile(const std::string& plan_filepath);
  
  std::vector<std::unique_ptr<TaskGraph>> ordered_task_gphs_;

};

void Compiler::ConstForEachChainNode(std::function<void(const ChainNode*)> func) {
  for (const auto& task_gph : ordered_task_gphs_) {
    task_gph->chain_gph()->ConstForEachNode([&](const ChainNode* chain) {
      func(chain);
    });
  }
}

void Compiler::ConstForEachStageNode(std::function<void(const StageNode*)> func) {
  for (const auto& task_gph : ordered_task_gphs_) {
    task_gph->stage_gph()->ConstForEachNode([&](const StageNode* stage) {
      func(stage);
    });
  }
}

void Compiler::ForEachTaskNode(std::function<void(TaskNode*)> func) {
  for (const auto& task_gph : ordered_task_gphs_) {
    task_gph->ForEachNode([&](TaskNode* task) {
      func(task);
    });
  }
}

// TODO: inference "piece_size" and "register_num for each register_desc"
void Compiler::Compile(const JobConf& job_conf,
                       const std::string& plan_filepath) {
  JobDesc::Singleton().InitFromJobConf(job_conf);
  JobDesc::Singleton().set_piece_size(50);
  IDMgr::Singleton().InitFromResource(JobDesc::Singleton().resource());

  BuildGraphs();
  InferShape4Regsts();
  EraseMeaningLessRegsts();
  GenPlanFile(plan_filepath);
}

void Compiler::BuildGraphs() {
  ordered_task_gphs_.clear();
  // data graph
  LOG(INFO) << "Build DataTaskGraph...";
  auto data_task_gph = new DataTaskGraph(
        "data",
        JobDesc::Singleton().train_dlnet_conf(),
        JobDesc::Singleton().strategy(),
        JobDesc::Singleton().is_train());
  ordered_task_gphs_.emplace_back(data_task_gph);
  // construct data_chain2sorted_fw_comp_tasks
  HashMap<const ChainNode*, std::vector<CompTaskNode*>>
      data_chain2sorted_fw_comp_tasks;
  data_task_gph->ForEachNode([&](TaskNode* node) {
    auto fw_node = dynamic_cast<CompTaskNode*>(node);
    if (fw_node == nullptr || fw_node->IsBpNode()
                           || fw_node->IsLossNode()) { return; }
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
  ForEachTaskNode([](const TaskNode* node) {
    std::string file_path = DotDir() + "/exec/" + node->node_id_str() + ".dot";
    node->exec_gph().ToDotFile(file_path);
  });
}

void Compiler::BuildModelGraphs(
    const std::pair<const ChainNode*, std::vector<CompTaskNode*>>& pair) {
  if (pair.first->HasOpWithModelOrModelTmpBlob() == false) { return; } 
  std::string chain_tag = pair.first->op_vec().front()->op_name();
  str_replace(&chain_tag, '/', '_');
  const std::string dot_path_prefix = DotDir() + "/model/" + chain_tag + "_";
  ParallelPolicy policy = pair.first->parallel_desc()->policy();
  LOG(INFO) << "Build MdUpdtTaskGraph... for " << chain_tag;
  auto updt_gph = new MdUpdtTaskGraph(
      "md_updt_" + chain_tag,
      pair.first, pair.second, dot_path_prefix + "model_update_");
  ordered_task_gphs_.emplace_back(updt_gph);
  if (JobDesc::Singleton().is_train()) {
    LOG(INFO) << "Build MdSaveTaskGraph... for " << chain_tag;
    ChainNode* updt_chain = updt_gph->chain_gph()->SoleSinkNode();
    auto updt_tasks = updt_gph->SortedCompTasksInChain(updt_chain);
    if (policy == kDataParallel) { updt_tasks = {updt_tasks.front()}; }
    for (CompTaskNode* update_task : updt_tasks) {
      auto save_gph = new MdSaveTaskGraph(
          "md_save_" + update_task->node_id_str(),
          update_task,
          dot_path_prefix + "model_save_" + update_task->node_id_str() + "_");
      ordered_task_gphs_.emplace_back(save_gph);
    }
  }
}

void Compiler::InferShape4Regsts() {
  for (auto& task_gph : ordered_task_gphs_) {
    LOG(INFO) << "InferShape... for " << task_gph->name();
    task_gph->InferShapeOfBlobsInProducedRegsts();
  }
}

void Compiler::EraseMeaningLessRegsts() {
  ForEachTaskNode([](TaskNode* task_node) {
    task_node->EraseZeroSizeBlobInProducedRegsts();
    task_node->EraseProducedEmptyRegsts();
  });
}

void Compiler::GenPlanFile(const std::string& plan_filepath) {
  Plan plan;
  ForEachTaskNode([&plan](const TaskNode* node) {
    if (!node->IsMeaningLess()) {
      node->ToProto(plan.mutable_task()->Add());
    }
  });
  OpMgr::Singleton().AllOpToProto(plan.mutable_op());
  JobDesc::Singleton().ToProto(plan.mutable_job_desc());
  ConstForEachChainNode([&plan](const ChainNode* node) {
    for (std::shared_ptr<const Operator> op : node->op_vec()) {
      CHECK(plan.mutable_op_name2device_type()->insert(
          {op->op_name(), node->parallel_desc()->device_type()}).second);
    }
  });
  ConstForEachStageNode([&plan](const StageNode* node) {
    auto pbmap = plan.mutable_machine_id2op_name_set();
    for (std::shared_ptr<const Operator> op : node->chain_node()->op_vec()) {
      (*pbmap)[node->machine_id()].add_op_name(op->op_name());
    }
  });
  PrintProtoToTextFile(plan, plan_filepath);
}

} // namespace oneflow

DEFINE_string(job_conf_filepath, "", "");
DEFINE_string(plan_filepath, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Compiler Starting Up...";
  oneflow::JobConf job_conf;
  oneflow::ParseProtoFromTextFile(FLAGS_job_conf_filepath, &job_conf);
  oneflow::Compiler::Singleton().Compile(job_conf, FLAGS_plan_filepath);
  LOG(INFO) << "Compiler Shutting Down...";
  return 0;
}
