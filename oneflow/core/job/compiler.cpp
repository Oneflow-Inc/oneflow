#include "oneflow/core/job/compiler.h"

namespace oneflow {

namespace {

std::map<TaskType, std::pair<std::string, std::string>> task_type2dot_type = {
    {kInvalid, {"plaintext", ""}},   {kForward, {"house", ""}},
    {kBackward, {"invhouse", ""}},   {kSource, {"folder", ""}},
    {kLoss, {"ellipse", ""}},        {kLossAcc, {"ellipse", "filled"}},
    {kLossPrint, {"signature", ""}}, {kMdUpdt, {"box", "rounded"}},
    {kMdSave, {"folder", "filled"}}, {kMdDiffAcc, {"box", "rounded,filled"}},
    {kCopyHd, {"note", ""}},         {kCopyCommNet, {"tab", ""}},
    {kBoxing, {"box3d", ""}},        {kPrint, {"signature", ""}},
};

void ToDotFile(const Plan& plan, const std::string& filepath) {
  PersistentOutStream out_stream(LocalFS(), filepath);
  out_stream << "digraph {\n";
  HashSet<int64_t> regst_desc_ids;
  for (const TaskProto& task_proto : plan.task()) {
    out_stream << "task" << std::to_string(task_proto.task_id()) << "[label=\""
               << std::to_string(task_proto.task_id())
               << "\\nthrd_id:" << std::to_string(task_proto.thrd_id())
               << "\\nparallel_id:"
               << std::to_string(task_proto.parallel_ctx().parallel_id())
               << "\", shape="
               << task_type2dot_type[task_proto.task_type()].first;
    if (task_type2dot_type[task_proto.task_type()].second != "")
      out_stream << ", style=\""
                 << task_type2dot_type[task_proto.task_type()].second << "\"";
    out_stream << "];\n";
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
      out_stream << "task" << std::to_string(task_proto.task_id())
                 << "->regst_desc"
                 << std::to_string(pair.second.regst_desc_id()) << "[label=\""
                 << pair.first << "\"];\n";
    }
    for (const auto& pair : task_proto.consumed_regst_desc_id()) {
      out_stream << "regst_desc" << std::to_string(pair.second) << "->task"
                 << std::to_string(task_proto.task_id()) << "[label=\""
                 << pair.first << "\"];\n";
    }
  }
  out_stream << "}\n";
}
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
