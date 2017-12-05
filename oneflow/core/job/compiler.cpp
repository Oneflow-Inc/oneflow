#include "oneflow/core/job/compiler.h"

namespace oneflow {

namespace {

void ToDotFile(const Plan& plan, const std::string& filepath) {}

}  // namespace

Plan Compiler::Compile() {
  LogicalGraph::NewSingleton();
  Plan plan = DoCompile();
  LogicalGraph::DeleteSingleton();
  return plan;
}

Plan Compiler::DoCompile() {
  auto chain_gph = of_make_unique<ChainGraph>(JobDesc::Singleton()->IsTrain());
  auto task_gph = of_make_unique<TaskGraph>(std::move(chain_gph));
  using std::placeholders::_1;
  task_gph->ForEachNode(std::bind(&TaskNode::ProduceAllRegstsAndBindEdges, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::ConsumeAllRegsts, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::Build, _1),
                        std::bind(&TaskNode::IsReadyForBuild, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::EraseEmptyProducedRegst, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::InferMemCaseOfProducedRegst, _1));
  Plan plan;
  task_gph->ForEachNode([&](TaskNode* task_node) {
    if (task_node->IsMeaningLess()) { return; }
    task_node->ToProto(plan.mutable_task()->Add());
  });
  plan.set_total_mbn_num(LogicalGraph::Singleton()->total_mbn_num());
  ToDotFile(plan, JoinPath(LogDir(), "/dot/plan.dot"));
  return plan;
}

}  // namespace oneflow
