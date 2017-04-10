#ifndef ONEFLOW_COMMON_ID_MANAGER_H_
#define ONEFLOW_COMMON_ID_MANAGER_H_

#include "common/util.h"
#include "job/resource.pb.h"

namespace oneflow {

// Glo  : Global
// Phy  : Physical
// Loc  : Local
// Thrd : Thread

using DeviceGloId = int32_t;
using DevicePhyId = int32_t;

using MachineId = int32_t;

using ThrdGloId = int32_t;
using ThrdLocId = int32_t;

class IDManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDManager);
  ~IDManager() = default;

  static IDManager& Singleton() {
    static IDManager obj;
    return obj;
  }

  void Init(const Resource& resource) {
    TODO();
  }

  ThrdLocId ThrdLocId4DevicePhyId(DevicePhyId) const {
    TODO();
  }
  ThrdLocId DataThrdLocId() const {
    TODO();
  }
  ThrdLocId BoxingThrdLocId() const {
    TODO();
  }

  int32_t NewNodeId();
  int32_t NewEdgeId();

 private:
  IDManager() = default;

};

} // namespace oneflow

#endif // ONEFLOW_COMMON_ID_MANAGER_H_
