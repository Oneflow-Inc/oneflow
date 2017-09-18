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

void TestDemo() {
  const auto& sfp = ScheduleFactoryConfigure::Provider("demo");
  const auto& sgraph_factory = sfp.sgraph_factory();
  const auto& analyzer_factory = sfp.utilization_analyzer_factory();
  const auto& session_factory = sfp.session_factory();
  const auto& engine_factory = sfp.schedule_engine_factory();
  const auto& allocator_factory = sfp.allocator_factory();
  const auto& validator_factory = sfp.validator_factory();

  const Plan* plan = nullptr;
  std::unique_ptr<SGraph> sgraph = sgraph_factory.CreateSGraph(*plan);
  std::unique_ptr<UtilizationAnalyzer> analyzer =
      analyzer_factory.CreateUtilizationAnalyzer(*sgraph);
  std::unique_ptr<UtilizationGraph> ugraph =
      analyzer->CreateUtilizationGraph("");
  std::unique_ptr<Session> sess =
      session_factory.CreateSession(*sgraph, *ugraph);
  auto schedule_engine = engine_factory.CreateScheduleEngine(*sess);
  std::unique_ptr<Allocator> allocator = allocator_factory.CreateAllocator();
  std::unique_ptr<Validator> validator = validator_factory.CreateValidator();

  std::unique_ptr<Schedule> schedule = schedule_engine->StaticSchedule();
  std::cout << "max-interval: " << schedule->max_interval() << std::endl;
  bool is_allocation_valid = validator->ValidateAllocation(*schedule);
  std::cout << "allocation is " << (is_allocation_valid ? "" : "NOT ")
            << "optimal" << std::endl;
  bool is_memory_valid = validator->ValidateMemory(*schedule);
  std::cout << "memory is " << (is_memory_valid ? "" : "NOT ") << "valid"
            << std::endl;
}

std::unique_ptr<Plan> LoadPlan(const std::string& file) {
  auto plan = of_make_unique<Plan>();
  ParseProtoFromTextFile(file, &*plan);
  return std::move(plan);
}

void TestPlan() {
  std::string conf = "default";
  //	std::string conf = "simulator_schedule_engine";
  //	std::string conf = "small_batch_num";
  //	std::string conf = "demo";
  const auto& sfp = ScheduleFactoryConfigure::Provider(conf);
  const auto& allocator_factory = sfp.allocator_factory();
  std::unique_ptr<Allocator> allocator = allocator_factory.CreateAllocator();
  const auto& sgraph_factory = sfp.sgraph_factory();
  const auto& analyzer_factory = sfp.utilization_analyzer_factory();
  const auto& session_factory = sfp.session_factory();
  const auto& validator_factory = sfp.validator_factory();
  std::unique_ptr<Validator> validator = validator_factory.CreateValidator();

  std::unique_ptr<Plan> plan = LoadPlan(FLAGS_plan);
  std::unique_ptr<SGraph> sgraph = sgraph_factory.CreateSGraph(*plan);
  std::ofstream(FLAGS_dot_dir + "/sgraph.dot", std::ofstream::out)
      << sgraph->ToDotString();
  std::unique_ptr<UtilizationAnalyzer> analyzer =
      analyzer_factory.CreateUtilizationAnalyzer(*sgraph);
  std::unique_ptr<UtilizationGraph> ugraph =
      analyzer->CreateUtilizationGraph("");
  validator->ValidateSGraph(*sgraph);
  std::unique_ptr<Session> session =
      session_factory.CreateSession(*sgraph, *ugraph);
  std::unique_ptr<Schedule> schedule =
      allocator->MemoryLimitedStaticSchedule(*session);
  schedule->PrintRegstNum();
  //  schedule->PrintSchedule();
  bool is_optimal = validator->ValidateAllocation(*schedule);
  std::cout << "allocation is " << (is_optimal ? "" : "NOT ") << "optimal"
            << std::endl;
}

void TestUtilizationGraph() {
  std::unique_ptr<Plan> plan = LoadPlan(FLAGS_plan);
  PlanSGraph sgraph(*plan);
  Validator validator(ScheduleFactoryConfigure::Provider("default"));
  validator.ValidateSGraph(sgraph);
  Simulator simulator;
  std::unique_ptr<UtilizationEventPackageProto> event_package =
      simulator.Run(sgraph);
  //  PrintProtoToTextFile(*event_package, "/tmp/a.proto");
  UtilizationAnalyzer analyzer(sgraph);
  std::unique_ptr<UtilizationGraph> ugraph = analyzer.Analyze(*event_package);
  Visualization visual;
  std::ofstream(FLAGS_dot_dir + "/ugraph.dot", std::ofstream::out)
      << visual.UGraph2DotString(*ugraph);
}

}  // namespace
}  // namespace schedule
}  // namespace oneflow

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  //  oneflow::schedule::TestDemo();
  //  oneflow::schedule::TestPlan();
  oneflow::schedule::TestUtilizationGraph();
  return 0;
}
