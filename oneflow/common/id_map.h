#ifndef ONEFLOW_COMMON_ID_MAP_H_
#define ONEFLOW_COMMON_ID_MAP_H_

#include "common/util.h"
#include "job/resource.pb.h"

namespace oneflow {

using DeviceGlobalId = int32_t;
using DevicePhysicalId = int32_t;

using MachineId = int32_t;

using ThreadGlobalId = int32_t;
using ThreadLocalId = int32_t;

class IDMap {
 public:
  DISALLOW_COPY_AND_MOVE(IDMap);
  IDMap() = default;
  ~IDMap() = default;

  void Init(const Resource& resource) {
    // TODO
  }
  ThreadLocalId ThreadLocalIdFromDevicePhysicalId(DevicePhysicalId) const {
    LOG(FATAL) << "TODO: implement it";
    return ThreadLocalId();
  }
  ThreadLocalId data_thread_local_id() const {
    LOG(FATAL) << "TODO: implement it";
    return ThreadLocalId();
  }
  ThreadLocalId boxing_thread_local_id() const {
    LOG(FATAL) << "TODO: implement it";
    return ThreadLocalId();
  }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_COMMON_ID_MAP_H_
