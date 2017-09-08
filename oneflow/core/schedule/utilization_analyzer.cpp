#include "oneflow/core/schedule/utilization_analyzer.h"
#include "oneflow/core/schedule/utilization_util.h"

namespace oneflow {
namespace schedule {

std::unique_ptr<UtilizationGraph> UtilizationAnalyzer::Analyze(
    const UtilizationPackageProto& utilization_package) const {
  auto ugraph = of_make_unique<UtilizationGraph>(sgraph());
  Analyze(ugraph.get());
  return ugraph;
}

std::unique_ptr<UtilizationGraph> UtilizationAnalyzer::Analyze(
    const UtilizationEventPackageProto& event_package) const {
  UtilizationPackageProto utilization_package;
  GetUtilizationPackageFromEvent(event_package, &utilization_package);
  return Analyze(utilization_package);
}

void UtilizationAnalyzer::Analyze(UtilizationGraph* ugraph) const {
  ugraph->ForEachUtilization(
      [&](Utilization* utilization) { utilization->Reduce(*ugraph); });
}

void UtilizationAnalyzer::ApplyUtilizationPackageProto(
    const UtilizationPackageProto& utilization_package,
    UtilizationGraph* ugraph) const {
  for (int uid = 0; uid < utilization_package.utilization_size(); ++uid) {
    const auto& utilization = utilization_package.utilization(uid);
    ApplyLeafUtilizationProto(utilization, ugraph);
  }
}

void UtilizationAnalyzer::GetUtilizationPackageFromEvent(
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

void UtilizationAnalyzer::ApplyLeafUtilizationProto(
    const UtilizationProto& utilization_proto, UtilizationGraph* graph) const {
  switch (utilization_proto.resource_type_case()) {
    case UtilizationProto::kTaskStreamResource:
      return ApplyTaskStreamUtilizationProto(utilization_proto, graph);
    case UtilizationProto::kRegstResource:
      return ApplyRegstUtilizationProto(utilization_proto, graph);
    default: UNEXPECTED_RUN();
  }
}

void UtilizationAnalyzer::ApplyTaskStreamUtilizationProto(
    const UtilizationProto& utilization_proto, UtilizationGraph* graph) const {
  CHECK(utilization_proto.has_task_stream_resource());
  std::string name = TaskStreamUtilization::MakeUniqueName(utilization_proto);
  auto tsu = graph->task_stream_mgr().Find(name);
  if (!tsu) {
    tsu = graph->mut_task_stream_mgr()->Create(utilization_proto);
    tsu->CreateAscendantIfNotFound(graph, tsu);
  }
  auto ptr = new UtilizationProto(utilization_proto);
  graph->mut_utilization_proto_store()->emplace(
      ptr, std::unique_ptr<UtilizationProto>(ptr));
  tsu->mut_raw_protos()->push_back(ptr);
  graph->c2leaf_arc_mgr().Input(tsu, [&](ComputationUtilization* cu) {
    cu->mut_raw_protos()->push_back(ptr);
  });
}

void UtilizationAnalyzer::ApplyRegstUtilizationProto(
    const UtilizationProto& utilization_proto, UtilizationGraph* graph) const {
  CHECK(utilization_proto.has_regst_resource());
  std::string name = RegstUtilization::MakeUniqueName(utilization_proto);
  auto ru = graph->regst_mgr().Find(name);
  if (!ru) {
    ru = graph->mut_regst_mgr()->Create(utilization_proto);
    ru->CreateAscendantIfNotFound(graph, ru);
  }
  auto ptr = new UtilizationProto(utilization_proto);
  graph->mut_utilization_proto_store()->emplace(
      ptr, std::unique_ptr<UtilizationProto>(ptr));
  ru->mut_raw_protos()->push_back(ptr);
  graph->m2leaf_arc_mgr().Input(
      ru, [&](MemoryUtilization* mu) { mu->mut_raw_protos()->push_back(ptr); });
}

}  // namespace schedule
}  // namespace oneflow
