#include "gflags/gflags.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/mem_info.h"
#include "oneflow/core/schedule/schedule_factory_configure.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

namespace {

void TestDemo() {
  auto sfp = ScheduleFactoryConfigure::Provider("demo");
  const auto& sgraph_factory = sfp->sgraph_factory();
  const auto& analyzer_factory = sfp->utilization_analyzer_factory();
  const auto& session_factory = sfp->session_factory();
  const auto& engine_factory = sfp->schedule_engine_factory();
  const auto& allocator_factory = sfp->allocator_factory();
  const auto& validator_factory = sfp->validator_factory();

  Plan* plan = nullptr;
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

void TestPlan(const std::string& file, const std::string& dot_file,
              const std::string& this_machine_name) {
  MemInfo::Singleton()->set_this_machine_name(this_machine_name);
  std::string conf = "default";
  //	std::string conf = "simulator_schedule_engine";
  //	std::string conf = "small_batch_num";
  //	std::string conf = "demo";
  auto sfp = ScheduleFactoryConfigure::Provider(conf);
  const auto& allocator_factory = sfp->allocator_factory();
  std::unique_ptr<Allocator> allocator = allocator_factory.CreateAllocator();
  const auto& sgraph_factory = sfp->sgraph_factory();
  const auto& analyzer_factory = sfp->utilization_analyzer_factory();
  const auto& session_factory = sfp->session_factory();
  const auto& validator_factory = sfp->validator_factory();
  std::unique_ptr<Validator> validator = validator_factory.CreateValidator();

  std::unique_ptr<Plan> plan = LoadPlan(file);
  std::unique_ptr<SGraph> sgraph = sgraph_factory.CreateSGraph(*plan);
  std::ofstream(dot_file, std::ofstream::out) << sgraph->ToDotString();
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

}  // namespace
}  // namespace schedule
}  // namespace oneflow

DEFINE_string(this_machine_name, "", "");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  //  oneflow::schedule::TestDemo();
  std::string dot_file = "/tmp/a.dot";
  if (argc > 2) { dot_file = argv[2]; }
  if (argc > 1) {
    std::string plan_file = argv[1];
    oneflow::schedule::TestPlan(plan_file, dot_file, FLAGS_this_machine_name);
  }
  return 0;
}
