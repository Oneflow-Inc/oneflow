#ifndef ONEFLOW_CORE_SCHEDULE_SIMULATOR_UTILIZATION_ANALYZER_H_
#define ONEFLOW_CORE_SCHEDULE_SIMULATOR_UTILIZATION_ANALYZER_H_

#include "oneflow/core/schedule/session.h"
#include "oneflow/core/schedule/simulator_schedule_engine.h"
#include "oneflow/core/schedule/utilization_analyzer.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

class SimulatorUtilizationAnalyzer : public UtilizationAnalyzer {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SimulatorUtilizationAnalyzer);
  explicit SimulatorUtilizationAnalyzer(const SGraph& sgraph)
      : UtilizationAnalyzer(sgraph) {}
  ~SimulatorUtilizationAnalyzer() = default;

 protected:
  std::unique_ptr<DeviceInfoProto> ParseDeviceInfoProto(
      const std::string& log_file) const {
    UtilizationGraph ugraph(*sgraph());
    Session session(*sgraph(), ugraph);
    SimulatorScheduleEngine engine(session);
    auto get_regst_num = [](uint64_t) -> uint32_t { return 3u; };
    auto schedule = engine.StaticSchedule(get_regst_num);
    auto simulator_schedule = dynamic_cast<SimulatorSchedule*>(schedule.get());
    CHECK(simulator_schedule);
    return simulator_schedule->move_device_info_proto();
  }
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_CORE_SCHEDULE_SIMULATOR_UTILIZATION_ANALYZER_H_
