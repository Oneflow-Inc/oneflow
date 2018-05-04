#include "oneflow/core/job/compiler.h"

namespace oneflow {

namespace {

void ToDotFile(const Plan& plan, const std::string& filepath) {
  PersistentOutStream out_stream(LocalFS(), filepath);
  out_stream << "digraph {\n";
  HashSet<int64_t> regst_desc_ids;
  for (const TaskProto& task_proto : plan.task()) {
    out_stream << "task" << std::to_string(task_proto.task_id()) << "[label=\""
               << std::to_string(task_proto.task_id()) << "\\n"
               << std::to_string(task_proto.machine_id()) << ":"
               << std::to_string(task_proto.thrd_id()) << ":"
               << std::to_string(task_proto.parallel_ctx().parallel_id())
               << "\", shape=ellipse, style=\"rounded,filled\", "
                  "colorscheme=set312, color="
               << task_type2color.at(task_proto.task_type()) << "];\n";
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
      out_stream << "task" << std::to_string(task_proto.task_id()) << "->regst_desc"
                 << std::to_string(pair.second.regst_desc_id()) << "[label=\"" << pair.first
                 << "\"];\n";
    }
    for (const auto& pair : task_proto.consumed_regst_desc_id()) {
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        out_stream << "regst_desc" << std::to_string(regst_desc_id) << "->task"
                   << std::to_string(task_proto.task_id()) << "[label=\"" << pair.first << "\"];\n";
      }
    }
  }
  out_stream << "}\n";
}
}  // namespace

Plan Compiler::Compile() {
  Plan plan = DoCompile();
  return plan;
}

Plan Compiler::DoCompile() {
  auto logical_gph = of_make_unique<LogicalGraph>(Global<JobDesc>::Get()->IsTrain());
  int64_t total_mbn_num = logical_gph->total_mbn_num();
  auto task_gph = of_make_unique<TaskGraph>(std::move(logical_gph));
  using std::placeholders::_1;
  task_gph->ForEachNode(std::bind(&TaskNode::ProduceAllRegstsAndBindEdges, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::ConsumeAllRegsts, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::PinConsumedRegst, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::Build, _1), std::bind(&TaskNode::IsReadyForBuild, _1));
  task_gph->ForEachNode(std::bind(&TaskNode::EraseEmptyProducedRegst, _1));
  Plan plan;
  task_gph->ForEachNode([&](TaskNode* task_node) {
    if (task_node->IsMeaningLess()) { return; }
    task_node->ToProto(plan.mutable_task()->Add());
  });
  plan.set_total_mbn_num(total_mbn_num);
  ToDotFile(plan, JoinPath(LogDir(), "/dot/plan.dot"));
  return plan;
}

}  // namespace oneflow
