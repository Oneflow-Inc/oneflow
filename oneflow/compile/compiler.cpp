#include "gflags/gflags.h"
#include "glog/logging.h"
#include "common/id_manager.h"
#include "common/protobuf.h"
#include "graph/model_load_task_graph.h"
#include "graph/model_save_task_graph.h"
#include "graph/model_update_task_graph.h"
#include "graph/data_task_graph.h"
#include "register/register_desc.h"
#include "conf/job_conf.pb.h"
#include "common/ofelf.pb.h"

namespace oneflow {

class Compiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Compiler);
  ~Compiler() = default;

  static Compiler& Singleton() {
    static Compiler obj;
    return obj;
  }

  void Compile(const JobConf& job_conf, const std::string& elf_filepath);

 private:
  Compiler() = default;
  void ForEachChainNode(std::function<void(ChainNode*)> func);
  void ForEachStageNode(std::function<void(StageNode*)> func);
  void ForEachTaskNode(std::function<void(TaskNode*)> func);
  
  void BuildGraphs();
  void InferShape4Regsts();
  void EraseMeaningLessNodesAndRegsts();
  
  std::vector<std::unique_ptr<TaskGraph>> ordered_task_gphs_;

};

void Compiler::ForEachChainNode(std::function<void(ChainNode*)> func) {
  for (const auto& task_gph : ordered_task_gphs_) {
    for (const auto& chain_node : task_gph->chain_gph()->nodes()) {
      func(chain_node.get());
    }
  }
}

void Compiler::ForEachStageNode(std::function<void(StageNode*)> func) {
  for (const auto& task_gph : ordered_task_gphs_) {
    for (const auto& stage_node : task_gph->stage_gph()->nodes()) {
      func(stage_node.get());
    }
  }
}

void Compiler::ForEachTaskNode(std::function<void(TaskNode*)> func) {
  for (const auto& task_gph : ordered_task_gphs_) {
    for (const auto& task_node : task_gph->nodes()) {
      func(task_node.get());
    }
  }
}

// TODO: inference "piece_size" and "register_num for each register_desc"
void Compiler::Compile(const JobConf& job_conf,
                       const std::string& elf_filepath) {
  JobDesc::Singleton().InitFromJobConf(job_conf);
  IDMgr::Singleton().InitFromResource(JobDesc::Singleton().resource());

  BuildGraphs();
  InferShape4Regsts();
  EraseMeaningLessNodesAndRegsts();
  OfElf elf;
  ForEachTaskNode([&elf](TaskNode* node) {
    if (!node->produced_regst_descs().empty()) {
      node->ToProto(elf.mutable_task()->Add());
    }
  });
  OpMgr::Singleton().AllOpToProto(elf.mutable_op());
  JobDesc::Singleton().ToProto(elf.mutable_job_desc());
  ForEachChainNode([&elf](ChainNode* node) {
    for (std::shared_ptr<const Operator> op : node->op_vec()) {
      CHECK(elf.mutable_op_name2device_type()->insert(
          {op->op_name(), node->parallel_desc()->device_type()}).second);
    }
  });
  ForEachStageNode([&elf](StageNode* node) {
    auto pbmap = elf.mutable_machine_id2op_name_set();
    for (std::shared_ptr<const Operator> op : node->chain_node()->op_vec()) {
      (*pbmap)[node->machine_id()].add_op_name(op->op_name());
    }
  });
  PrintProtoToTextFile(elf, elf_filepath);
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
  for (const auto& node : data_task_gph->nodes()) {
    auto fw_node = dynamic_cast<CompTaskNode*>(node.get());
    if (fw_node == nullptr || fw_node->IsBpNode()
                           || fw_node->IsLossNode()) { continue; }
    data_chain2sorted_fw_comp_tasks[fw_node->chain_node()].push_back(fw_node);
  }
  for (auto& pair : data_chain2sorted_fw_comp_tasks) {
    SortByParallelId(&(pair.second));
  }
  // model graph
  for (const auto& pair : data_chain2sorted_fw_comp_tasks) {
    if (pair.first->HasOpWithModelOrModelTmpBlob() == false) { continue; } 
    std::string chain_tag = pair.first->op_vec().front()->op_name();
    str_replace(&chain_tag, '/', '_');
    const std::string dot_path_prefix = DotDir() + "/model/" + chain_tag + "_";
    ParallelPolicy policy = pair.first->parallel_desc()->policy();
    LOG(INFO) << "Build MdUpdtTaskGraph... for " << chain_tag;
    auto updt_gph = new MdUpdtTaskGraph(
        "md_updt_" + chain_tag,
        pair.first, pair.second, dot_path_prefix + "model_update_");
    ChainNode* updt_chain = updt_gph->chain_gph()->SoleSinkNode();
    auto sorted_updt_tasks = updt_gph->SortedCompTasksInChain(updt_chain);
    HashMap<uint64_t, CompTaskNode*> parallel_id2updt_task;
    for (CompTaskNode* update_task : sorted_updt_tasks) {
      CHECK(parallel_id2updt_task.emplace(
            update_task->parallel_id(), update_task).second);
    }
    ordered_task_gphs_.emplace_back(updt_gph);
    LOG(INFO) << "Build MdLoadTaskGraph... for " << chain_tag;
    auto load_gph = new MdLoadTaskGraph(
        "md_load_" + chain_tag,
        updt_chain, parallel_id2updt_task, policy,
        dot_path_prefix + "model_load_");
    ordered_task_gphs_.emplace_back(load_gph);
    if (JobDesc::Singleton().is_train()) {
      LOG(INFO) << "Build MdSaveTaskGraph... for " << chain_tag;
      auto save_gph = new MdSaveTaskGraph(
          "md_save_" + chain_tag,
          updt_chain, parallel_id2updt_task, policy,
          dot_path_prefix + "model_save_");
      ordered_task_gphs_.emplace_back(save_gph);
    }
  }
  // all exec_graph 2 dot
  ForEachTaskNode([](TaskNode* node) {
    std::string file_path = DotDir() + "/exec/" + node->node_id_str() + ".dot";
    node->exec_gph().ToDotFile(file_path);
  });
}

void Compiler::InferShape4Regsts() {
  for (auto& task_gph : ordered_task_gphs_) {
    LOG(INFO) << "InferShape... for " << task_gph->name();
    task_gph->InferShapeOfBlobsInProducedRegsts();
  }
}

void Compiler::EraseMeaningLessNodesAndRegsts() {
  ForEachTaskNode([](TaskNode* task_node) {
    task_node->EraseZeroSizeBlobInProducedRegsts();
    task_node->EraseProducedEmptyRegsts();
  });
}

} // namespace oneflow

DEFINE_string(job_conf_filepath, "", "");
DEFINE_string(elf_filepath, "", "");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Compiler Starting Up...";
  oneflow::JobConf job_conf;
  oneflow::ParseProtoFromTextFile(FLAGS_job_conf_filepath, &job_conf);
  oneflow::Compiler::Singleton().Compile(job_conf, FLAGS_elf_filepath);
  LOG(INFO) << "Compiler Shutting Down...";
  return 0;
}
