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
DEFINE_string(dot_dir, "./tmp", "dot file directory");

namespace oneflow {
namespace schedule {

namespace {

std::unique_ptr<Plan> LoadPlan(const std::string& file) {
  auto plan = of_make_unique<Plan>();
  ParseProtoFromTextFile(file, &*plan);
  return std::move(plan);
}

void Simulate() {
  std::unique_ptr<Plan> plan = LoadPlan(FLAGS_plan);
  PlanSGraph sgraph(*plan);
  Validator validator(ScheduleFactoryConfigure::Provider("default"));
  validator.ValidateSGraph(sgraph);
  Simulator simulator;
  std::unique_ptr<UtilizationEventPackageProto> event_package =
      simulator.Run(sgraph);
  PrintProtoToTextFile(*event_package, FLAGS_dot_dir + "/event-package.proto");
}

}  // namespace
}  // namespace schedule
}  // namespace oneflow

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::schedule::Simulate();
  return 0;
}
