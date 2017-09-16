#include "oneflow/core/schedule/utilization_analyzer.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/schedule/utilization_util.h"

namespace oneflow {
namespace schedule {

void UtilizationAnalyzer::ForEachDeviceMemory(
    const std::function<void(const std::string&, uint64_t)>& cb) const {
  //	TODO:
  //	cb(device_name0, memory_size0);
  //	cb(device_name1, memory_size1);
  //	cb(device_name2, memory_size2);
  //	and so on;
}

std::unique_ptr<UtilizationEventPackageProto>
UtilizationAnalyzer::ParseEventPackageProto(const std::string& log_file) const {
  auto event_package = of_make_unique<UtilizationEventPackageProto>();
  ParseProtoFromTextFile(log_file, event_package.get());
  return std::move(event_package);
}

std::unique_ptr<UtilizationGraph> UtilizationAnalyzer::CreateUtilizationGraph(
    std::string log_file) {
  auto event_package = ParseEventPackageProto(log_file);
  return Analyze(*event_package);
}

std::unique_ptr<UtilizationGraph> UtilizationAnalyzer::Analyze(
    const UtilizationPackageProto& utilization_package) const {
  auto ugraph = of_make_unique<UtilizationGraph>(sgraph());
  AddUtilizationPackageProto(utilization_package, ugraph.get());
  Analyze(ugraph.get());
  return ugraph;
}

std::unique_ptr<UtilizationGraph> UtilizationAnalyzer::Analyze(
    const UtilizationEventPackageProto& event_package) const {
  CHECK(event_package.event_size());
  UtilizationPackageProto utilization_package;
  GetUtilizationPackageFromEvent(event_package, &utilization_package);
  return Analyze(utilization_package);
}

void UtilizationAnalyzer::Analyze(UtilizationGraph* ugraph) const {
  ugraph->ForEachUtilization(
      [&](Utilization* utilization) { utilization->Reduce(*ugraph); });
}

void UtilizationAnalyzer::AddUtilizationPackageProto(
    const UtilizationPackageProto& utilization_package,
    UtilizationGraph* ugraph) const {
  for (int uid = 0; uid < utilization_package.utilization_size(); ++uid) {
    const auto& utilization = utilization_package.utilization(uid);
    AddLeafUtilizationProto(utilization, ugraph);
  }
}

void UtilizationAnalyzer::GetUtilizationPackageFromEvent(
    const UtilizationEventPackageProto& event_package,
    UtilizationPackageProto* utilization_package) const {
  std::unordered_map<std::string, std::list<const UtilizationEventProto*>>
      grouped_events;
  for (int eid = 0; eid < event_package.event_size(); ++eid) {
    const UtilizationEventProto& event = event_package.event(eid);
    std::string key = UtilizationUtil::GetUniqueName(event.resource());
    key += "-" + event.batch_id();
    grouped_events[key].push_back(&event);
  }

  for (const auto& pair : grouped_events) {
    PackUtilizationProto(pair.second, utilization_package);
  }
}

void UtilizationAnalyzer::PackUtilizationProto(
    const std::list<const UtilizationEventProto*>& event_pair,
    UtilizationPackageProto* utilization_package) const {
  CHECK(event_pair.size() == 2);
  auto start_event = event_pair.front();
  auto end_event = event_pair.back();
  CHECK(start_event->time() < end_event->time());
  CHECK(start_event->event_type() == kStartEvent);
  CHECK(end_event->event_type() == kEndEvent);
  UtilizationProto* u = utilization_package->add_utilization();
  *u->mutable_resource() = start_event->resource();
  u->set_utilization(static_cast<float>(1));
  u->set_start_at(start_event->time());
  u->set_end_at(end_event->time());
  u->set_start_batch_id(start_event->batch_id());
  u->set_end_batch_id(end_event->batch_id());
}

template<typename U>
void UtilizationAnalyzer::AddUtilizationProto(
    const UtilizationProto& utilization_proto, UtilizationGraph* graph) const {
  CHECK(U::resource_type_case
        == utilization_proto.resource().resource_type_case());
  auto utilization =
      graph->FindOrCreateUtilization(utilization_proto.resource());
  auto ptr = new UtilizationProto(utilization_proto);
  graph->mut_utilization_proto_store()->emplace(
      ptr, std::unique_ptr<UtilizationProto>(ptr));
  uint32_t path_node_count =
      graph->ForEachUtilizationInPath(utilization, [&](Utilization* u) {
        u->mut_raw_protos()->push_back(ptr);
        if (u != utilization) {
          graph->mut_inner2leaf_arc_mgr()->CreateIfNotFound(u, utilization);
        }
      });
  CHECK(path_node_count > 3);
}

void UtilizationAnalyzer::AddLeafUtilizationProto(
    const UtilizationProto& utilization_proto, UtilizationGraph* graph) const {
  switch (utilization_proto.resource().resource_type_case()) {
#define ADD_UTILIZATION_PROTO_ENTRY(class_name) \
  case class_name::resource_type_case:          \
    return AddUtilizationProto<class_name>(utilization_proto, graph);
    OF_PP_FOR_EACH_TUPLE(ADD_UTILIZATION_PROTO_ENTRY, UTILIZATION_EVENT_SEQ)
    default: UNEXPECTED_RUN();
  }
}

}  // namespace schedule
}  // namespace oneflow
