#include "oneflow/core/schedule/utilization_analyzer.h"
#include "oneflow/core/schedule/utilization_util.h"

namespace oneflow {
namespace schedule {

void UtilizationAnalyzer::CreateUtilizationGraph(
    const UtilizationPackageProto& utilization_package,
    UtilizationGraph* ugraph) const {}

void UtilizationAnalyzer::CreateUtilizationFromEvent(
    const UtilizationEventPackageProto& event_package,
    UtilizationPackageProto* utilization_package) const {
  std::unordered_map<std::string, std::list<const UtilizationEventProto*>>
      grouped_events;
  for (int eid = 0; eid < event_package.event_size(); ++eid) {
    const UtilizationEventProto& event = event_package.event(eid);
    std::string key = UtilizationUtil::CreateUniqueName(event);
    key += "," + event.batch_id();
    grouped_events[key].push_back(&event);
  }

  for (const auto& pair : grouped_events) {
    AddUtilizationProto(pair.second, utilization_package);
  }
}

void UtilizationAnalyzer::AddUtilizationProto(
    const std::list<const UtilizationEventProto*>& event_pair,
    UtilizationPackageProto* utilization_package) const {
  CHECK(event_pair.size() == 2);
  const auto* start_event = event_pair.front();
  const auto* end_event = event_pair.back();
  CHECK(start_event->time() < end_event->time());
  CHECK(start_event->event_type() == kStartEvent);
  CHECK(end_event->event_type() == kEndEvent);
  UtilizationProto* u = utilization_package->add_utilization();
  u->set_utilization(static_cast<float>(1));
  u->set_start_at(start_event->time());
  u->set_end_at(end_event->time());
  u->set_start_batch_id(start_event->batch_id());
  u->set_end_batch_id(end_event->batch_id());
  UtilizationUtil::SetResourceType(*start_event, u);
}

}  // namespace schedule
}  // namespace oneflow
