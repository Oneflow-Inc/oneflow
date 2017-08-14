/**
 * Copyright 2017 Xinqi Li
 */
#include <fstream>
#include <sstream>
#include <string>
#include "oneflow/core/schedule/schedule_factory_configure.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"

namespace oneflow {
namespace schedule {

void TestDemo() {
  auto sfp = ScheduleFactoryConfigure::Provider("demo");
  auto sgraph_factory = sfp->sgraph_factory();
  auto session_factory = sfp->session_factory();
  auto engine_factory = sfp->schedule_engine_factory();
  auto validator_factory = sfp->validator_factory();

  auto sgraph = sgraph_factory->CreateSGraph("demo");
  auto sess = session_factory->CreateSession(*sgraph);
  auto schedule_engine = engine_factory->CreateScheduleEngine(*sess);
  auto validator = validator_factory->CreateValidator();

  auto schedule = schedule_engine->StaticSchedule();
  std::cout << "max-interval: " << schedule->max_interval() << std::endl;
  auto is_allocation_valid = validator->ValidateAllocation(*schedule);
  std::cout << "allocation is " << (is_allocation_valid ? "" : "NOT ")
            << "optimal" << std::endl;
  auto is_memory_valid = validator->ValidateMemory(*schedule);
  std::cout << "memory is " << (is_memory_valid ? "" : "NOT ") << "valid"
            << std::endl;
}

}  // namespace schedule
}  // namespace oneflow

int main(int argc, char* argv[]) {
  oneflow::schedule::TestDemo();
  return 0;
}
