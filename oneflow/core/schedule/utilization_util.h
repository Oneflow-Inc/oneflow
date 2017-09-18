#ifndef ONEFLOW_SCORE_SCHEDULE_UTILIZATION_UTIL_H_
#define ONEFLOW_SCORE_SCHEDULE_UTILIZATION_UTIL_H_

#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/utilization.pb.h"

namespace oneflow {
namespace schedule {

class UtilizationGraph;

class UtilizationUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UtilizationUtil);
  UtilizationUtil() = delete;
  ~UtilizationUtil() = delete;

  static std::string GetUniqueName(const UtilizationResource& resource) {
    return GetUniqueName(resource, "-");
  }
  static std::string GetUniqueName(const UtilizationResource& resource,
                                   const std::string& sep);

  static void ForEachGrouped(
      const UtilizationResource& resource, const UtilizationGraph& ugraph,
      const std::function<void(const UtilizationResource&)>& cb);

 private:
  template<UtilizationResource::ResourceTypeCase resource_type_case>
  static std::string GetResourceUniqueName(const UtilizationResource& resource,
                                           const std::string& sep);

  template<UtilizationResource::ResourceTypeCase resource_type_case>
  static void ForEachGroupedResource(
      const UtilizationResource& resource, const UtilizationGraph& ugraph,
      const std::function<void(const UtilizationResource&)>& cb);
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_SCORE_SCHEDULE_UTILIZATION_UTIL_H_
