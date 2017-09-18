#include "oneflow/core/schedule/simulator.h"
#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

std::unique_ptr<UtilizationEventPackageProto> Simulator::Run(
    const SGraph& sgraph) const {
  UtilizationGraph ugraph(sgraph);
  Session session(sgraph, ugraph);
  SimulatorScheduleEngine engine(session);
  auto get_regst_num = [&](uint64_t id) -> uint32_t {
    const SRegstDesc* regst_desc = sgraph.node_mgr<SRegstDesc>().Find(id);
    CHECK(regst_desc);
    return regst_desc->origin_regst_count();
  };
  auto schedule = engine.StaticSchedule(get_regst_num);
  auto simulator_schedule = dynamic_cast<SimulatorSchedule*>(schedule.get());
  CHECK(simulator_schedule);
  return simulator_schedule->move_event_package_proto();
}

}  // namespace schedule
}  // namespace oneflow
