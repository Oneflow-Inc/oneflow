#ifndef ONEFLOW_CORE_COMPILER_COMPILER_H_
#define ONEFLOW_CORE_COMPILER_COMPILER_H_
#include "gflags/gflags.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/stage_graph.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.pb.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

namespace compiler {

class Compiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Compiler);
  ~Compiler() = default;

  OF_SINGLETON(Compiler);

  void Compile(const JobConf& job_conf, const std::string& plan_filepath);
  void Compile(const JobDescProto& job_conf, Plan* plan);

 private:
  Compiler() = default;
  void ConstForEachChainNode(std::function<void(const ChainNode*)> func);
  void ConstForEachStageNode(std::function<void(const StageNode*)> func);
  void ForEachTaskNode(std::function<void(TaskNode*)> func);

  void BuildGraphs();
  void BuildModelGraphs(
      const std::pair<const ChainNode*, std::vector<CompTaskNode*>>&);
  void InferShape4Regsts();
  void EraseMeaningLessRegsts();
  void GenPlanFile(const std::string& plan_filepath);
  void GenPlan(Plan* plan);
  void Plan2DotFile(const Plan& plan);

  std::vector<std::unique_ptr<TaskGraph>> ordered_task_gphs_;
};

}  // namespace compiler

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_COMPILER_H_
