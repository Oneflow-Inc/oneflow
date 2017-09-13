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
  explicit UtilizationAnalyzer(const SGraph& sgraph) : sgraph_(&sgraph) {}
  virtual ~UtilizationAnalyzer() = default;

  inline const SGraph* sgraph() const { return sgraph_; }

  std::unique_ptr<UtilizationGraph> CreateUtilizationGraph(
      std::string log_file);

 protected:
  virtual void ParseDeviceInfoProto(const std::string& log_file,
                                    DeviceInfoProto* device_info_proto) const;
  virtual std::unique_ptr<UtilizationGraph> Analyze(
      const DeviceInfoProto& dev_info_package) const;

 private:
  std::unique_ptr<UtilizationGraph> Analyze(
      const UtilizationPackageProto& utilization_package) const;

  void GetUtilizationPackageFromEvent(
      const DeviceInfoProto& event_package,
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
  const SGraph* sgraph_;
};

class EmptyUtilizationAnalyzer : public UtilizationAnalyzer {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmptyUtilizationAnalyzer);
  explicit EmptyUtilizationAnalyzer(const SGraph& sgraph)
      : UtilizationAnalyzer(sgraph) {}
  ~EmptyUtilizationAnalyzer() = default;

 protected:
  std::unique_ptr<UtilizationGraph> Analyze(
      const DeviceInfoProto& dev_info_package) const override {
    return of_make_unique<UtilizationGraph>(*sgraph());
  }
};

}  // namespace schedule
}  // namespace oneflow

#endif  // ONEFLOW_CORE_SCHEDULE_UTILIZATION_ANALYZER_H_
