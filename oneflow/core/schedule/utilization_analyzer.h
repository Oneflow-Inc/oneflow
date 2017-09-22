#ifndef ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_H_
#define ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/sgraph.h"
#include "oneflow/core/schedule/utilization.pb.h"
#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

class UtilizationAnalyzer {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UtilizationAnalyzer);
  UtilizationAnalyzer() = default;
  virtual ~UtilizationAnalyzer() = default;

  std::unique_ptr<UtilizationGraph> CreateUtilizationGraph(
      const SGraph& sgraph, const std::string& log_file);
  virtual std::unique_ptr<UtilizationGraph> Analyze(
      const SGraph& sgraph,
      const UtilizationEventPackageProto& dev_info_package) const;

 protected:
  virtual void ForEachDeviceMemory(
      const std::function<void(const std::string&, uint64_t)>& cb) const;
  virtual std::unique_ptr<UtilizationEventPackageProto> ParseEventPackageProto(
      const SGraph& sgraph, const std::string& log_file) const;

 private:
  std::unique_ptr<UtilizationGraph> Analyze(
      const SGraph& sgraph,
      const UtilizationPackageProto& utilization_package) const;

  void GetUtilizationPackageFromEvent(
      const UtilizationEventPackageProto& event_package,
      UtilizationPackageProto* utilization_package) const;
  void Analyze(UtilizationGraph* ugraph) const;
  void AddUtilizationPackageProto(
      const UtilizationPackageProto& utilization_package,
      UtilizationGraph* ugraph) const;
  void AddLeafUtilizationProto(const UtilizationProto& utilization_proto,
                               UtilizationGraph* graph) const;
  template<typename U>
  void AddUtilizationProto(const UtilizationProto& utilization_proto,
                           UtilizationGraph* graph) const;
  void PackUtilizationProto(
      const std::list<const UtilizationEventProto*>& event_pair,
      UtilizationPackageProto* utilization_package) const;
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_H_
