#ifndef ONEFLOW_COMMON_ID_MANAGER_H_
#define ONEFLOW_COMMON_ID_MANAGER_H_

#include "common/util.h"
#include "job/resource.pb.h"

namespace oneflow {

using DeviceGlobalId = int32_t;
using DevicePhysicalId = int32_t;

using MachineId = int32_t;

using ThreadGlobalId = int32_t;
using ThreadLocalId = int32_t;

class IDManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDManager);
  ~IDManager() = default;

  static IDManager& Singleton() {
    static IDManager obj;
    return obj;
  }

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
  IDManager() = default;

};

} // namespace oneflow

#endif // ONEFLOW_COMMON_ID_MANAGER_H_
