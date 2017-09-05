#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/schedule_factory_configure.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"

namespace oneflow {
namespace schedule {

namespace {

void TestDemo() {
  auto sfp = ScheduleFactoryConfigure::Provider("demo");
  auto sgraph_factory = sfp->sgraph_factory();
  auto session_factory = sfp->session_factory();
  auto engine_factory = sfp->schedule_engine_factory();
  auto allocator_factory = sfp->allocator_factory();
  auto validator_factory = sfp->validator_factory();

  Plan* plan = nullptr;
  std::unique_ptr<SGraph> sgraph = sgraph_factory->CreateSGraph(*plan);
  std::unique_ptr<Session> sess = session_factory->CreateSession(*sgraph);
  auto schedule_engine = engine_factory->CreateScheduleEngine(*sess);
  std::unique_ptr<Allocator> allocator = allocator_factory->CreateAllocator();
  std::unique_ptr<Validator> validator = validator_factory->CreateValidator();

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

void TestPlan(const std::string& file, const std::string& dot_file) {
  std::string conf = "default";
  //	std::string conf = "small_batch_num";
  //	std::string conf = "demo";
  auto sfp = ScheduleFactoryConfigure::Provider(conf);
  auto allocator_factory = sfp->allocator_factory();
  std::unique_ptr<Allocator> allocator = allocator_factory->CreateAllocator();
  auto sgraph_factory = sfp->sgraph_factory();
  auto session_factory = sfp->session_factory();
  auto validator_factory = sfp->validator_factory();
  std::unique_ptr<Validator> validator = validator_factory->CreateValidator();

  std::unique_ptr<Plan> plan = LoadPlan(file);
  std::unique_ptr<SGraph> sgraph = sgraph_factory->CreateSGraph(*plan);
  std::ofstream(dot_file, std::ofstream::out) << sgraph->ToDotString();
  validator->ValidateGraph(*sgraph);
  std::unique_ptr<Session> session = session_factory->CreateSession(*sgraph);
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

int main(int argc, char* argv[]) {
  //  oneflow::schedule::TestDemo();
  std::string dot_file = "/tmp/a.dot";
  if (argc > 2) { dot_file = argv[2]; }
  if (argc > 1) {
    std::string plan_file = argv[1];
    oneflow::schedule::TestPlan(plan_file, dot_file);
  }
  return 0;
}
