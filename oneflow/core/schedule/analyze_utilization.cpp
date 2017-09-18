#include "gflags/gflags.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/plan_sgraph.h"
#include "oneflow/core/schedule/schedule_factory_configure.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulator.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"
#include "oneflow/core/schedule/utilization_graph.h"
#include "oneflow/core/schedule/visualization.h"

DEFINE_string(plan, "", "plan file");
DEFINE_string(event_package, "", "utilization event package proto file");
DEFINE_string(dot_dir, "./tmp", "dot file directory");

namespace oneflow {
namespace schedule {

namespace {

std::unique_ptr<Plan> LoadPlan(const std::string& file) {
  auto plan = of_make_unique<Plan>();
  ParseProtoFromTextFile(file, &*plan);
  return std::move(plan);
}

void AnalyzeUtilization() {
  const auto& sfp = ScheduleFactoryConfigure::Provider("utilization");
  const auto& allocator_factory = sfp.allocator_factory();
  std::unique_ptr<Allocator> allocator = allocator_factory.CreateAllocator();
  const auto& sgraph_factory = sfp.sgraph_factory();
  const auto& analyzer_factory = sfp.utilization_analyzer_factory();
  std::unique_ptr<UtilizationAnalyzer> analyzer =
      analyzer_factory.CreateUtilizationAnalyzer();
  const auto& validator_factory = sfp.validator_factory();
  std::unique_ptr<Validator> validator = validator_factory.CreateValidator();

  Plan plan;
  ParseProtoFromTextFile(FLAGS_plan, &plan);
  std::unique_ptr<SGraph> sgraph = sgraph_factory.CreateSGraph(plan);
  validator->ValidateSGraph(*sgraph);
  std::ofstream(FLAGS_dot_dir + "/sgraph.dot", std::ofstream::out)
      << sgraph->ToDotString();
  std::unique_ptr<UtilizationGraph> ugraph =
      analyzer->CreateUtilizationGraph(*sgraph, FLAGS_event_package);
  Visualization visual;
  std::ofstream(FLAGS_dot_dir + "/ugraph.dot", std::ofstream::out)
      << visual.UGraph2DotString(*ugraph);
}

}  // namespace
}  // namespace schedule
}  // namespace oneflow

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::schedule::AnalyzeUtilization();
  return 0;
}
