#ifndef ONEFLOW_CORE_JOB_COMPILER_H_
#define ONEFLOW_CORE_JOB_COMPILER_H_

#include "oneflow/core/graph/data_comp_task_node.h"
#include "oneflow/core/graph/data_task_graph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class Compiler final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Compiler);
  ~Compiler() = default;

  OF_SINGLETON(Compiler);

  Plan Compile();

 private:
  Compiler() = default;
  void InitRelatedSingleton(const JobConf& job_conf);
  void ConstForEachChainNode(std::function<void(const ChainNode*)> func);
  void ConstForEachStageNode(std::function<void(const StageNode*)> func);
  void ForEachTaskNode(std::function<void(TaskNode*)> func);

  void BuildGraphs();
  void BuildModelGraphs(
      const std::pair<const ChainNode*, std::vector<CompTaskNode*>>&);
  void BuildLossGraph(
      const std::pair<const ChainNode*, std::vector<CompTaskNode*>>& pair);
  void InferBlobDesc4Regsts();
  void EraseMeaningLessRegsts();
  Plan GenPlanFile();
  void Plan2DotFile(const Plan& plan);

  std::vector<std::unique_ptr<TaskGraph>> ordered_task_gphs_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPILER_H_
