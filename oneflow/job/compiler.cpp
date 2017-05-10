#include "gflags/gflags.h"
#include "glog/logging.h"
#include "job/id_manager.h"
#include "common/proto_io.h"
#include "graph/model_load_task_graph.h"
#include "graph/model_save_task_graph.h"
#include "graph/model_update_task_graph.h"
#include "graph/data_task_graph.h"
#include "job/job_conf.pb.h"
#include "job/ofelf.pb.h"

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
  void RunFunc4EachTaskNode(std::function<void(TaskNode*)> func);
  
  void BuildGraphs();
  void RemoveRegstsWithoutBlob();
  void InferShape4Regsts();
  
  std::vector<std::unique_ptr<TaskGraph>> ordered_task_gphs_;

};

void Compiler::RunFunc4EachTaskNode(std::function<void(TaskNode*)> func) {
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
  RunFunc4EachTaskNode([](TaskNode* node) { node->RemoveRegstsWithoutBlob(); });
  InferShape4Regsts();
  OfElf elf;
  RunFunc4EachTaskNode([&elf](TaskNode* node) {
    node->ToProto(elf.mutable_task()->Add());
  });
  OpMgr::Singleton().AllOpToProto(elf.mutable_op());
  JobDesc::Singleton().ToProto(elf.mutable_job_desc());
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
        true);
  ordered_task_gphs_.emplace_back(data_task_gph);
  // construct data_chain2sorted_bp_comp_tasks
  HashMap<const ChainNode*, std::vector<CompTaskNode*>>
      data_chain2sorted_bp_comp_tasks;
  for (const auto& node : data_task_gph->nodes()) {
    auto bp_node = dynamic_cast<CompTaskNode*>(node.get());
    if (bp_node == nullptr || bp_node->IsFwNode()) { continue; }
    data_chain2sorted_bp_comp_tasks[bp_node->chain_node()].push_back(bp_node);
  }
  for (auto& pair : data_chain2sorted_bp_comp_tasks) {
    SortByParallelId(&(pair.second));
  }
  // model graph
  for (const auto& pair : data_chain2sorted_bp_comp_tasks) {
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
    LOG(INFO) << "Build MdLoadTaskGraph... for " << chain_tag;
    auto load_gph = new MdLoadTaskGraph(
        "md_load_" + chain_tag,
        updt_chain, parallel_id2updt_task, policy,
        dot_path_prefix + "model_load_");
    LOG(INFO) << "Build MdSaveTaskGraph... for " << chain_tag;
    auto save_gph = new MdSaveTaskGraph(
        "md_save_" + chain_tag,
        updt_chain, parallel_id2updt_task, policy,
        dot_path_prefix + "model_save_");
    ordered_task_gphs_.emplace_back(updt_gph);
    ordered_task_gphs_.emplace_back(load_gph);
    ordered_task_gphs_.emplace_back(save_gph);
  }
  // all exec_graph 2 dot
  RunFunc4EachTaskNode([](TaskNode* node) {
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
