#ifndef ONEFLOW_SCORE_SCHEDULE_UTILIZATION_UTIL_H_
#define ONEFLOW_SCORE_SCHEDULE_UTILIZATION_UTIL_H_

#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/schedule/utilization.pb.h"

namespace oneflow {
namespace schedule {

class UtilizationUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UtilizationUtil);
  UtilizationUtil() = delete;
  ~UtilizationUtil() = delete;

  static void SetResourceType(const UtilizationEventProto& event_proto,
                              UtilizationProto* utilization_proto);
  static std::string CreateUniqueName(const UtilizationEventProto& event_proto);

 private:
  static std::string CreateUniqueName(
      const TaskStreamResource& task_stream_res);
  static std::string CreateUniqueName(const RegstResource& regst_res);
};

}  // namespace schedule
}  // namespace oneflow
#endif  // ONEFLOW_SCORE_SCHEDULE_UTILIZATION_UTIL_H_
