/**
 * Copyright 2017 Xinqi Li
 */
#include <fstream>
#include <sstream>
#include <string>
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/schedule/schedule_factory_configure.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"
#include "oneflow/core/schedule/util.h"

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
  auto sgraph = sgraph_factory->CreateSGraph(*plan);
  auto sess = session_factory->CreateSession(*sgraph);
  auto schedule_engine = engine_factory->CreateScheduleEngine(*sess);
  auto allocator = allocator_factory->CreateAllocator();
  auto validator = validator_factory->CreateValidator();

  auto schedule = schedule_engine->StaticSchedule();
  //  auto schedule = allocator->MemoryLimitedStaticSchedule(*sess);
  std::cout << "max-interval: " << schedule->max_interval() << std::endl;
  auto is_allocation_valid = validator->ValidateAllocation(*schedule);
  std::cout << "allocation is " << (is_allocation_valid ? "" : "NOT ")
            << "optimal" << std::endl;
  auto is_memory_valid = validator->ValidateMemory(*schedule);
  std::cout << "memory is " << (is_memory_valid ? "" : "NOT ") << "valid"
            << std::endl;
}

std::unique_ptr<Plan> LoadPlan(const std::string& file) {
  auto plan = unique_ptr_new<Plan>();
  ParseProtoFromTextFile(file, &*plan);
  return std::move(plan);
}

void TestPlan(const std::string& file) {
  //  auto conf = "default";
  auto conf = "small-batch-num";
  //  auto conf = "demo";
  auto sfp = ScheduleFactoryConfigure::Provider(conf);
  auto allocator_factory = sfp->allocator_factory();
  auto allocator = allocator_factory->CreateAllocator();
  auto plan = LoadPlan(file);
  allocator->Allocate(&*plan);
}

}  // namespace
}  // namespace schedule
}  // namespace oneflow

int main(int argc, char* argv[]) {
  //  oneflow::schedule::TestDemo();
  if (argc > 1) { oneflow::schedule::TestPlan(std::string(argv[1])); }
  return 0;
}
