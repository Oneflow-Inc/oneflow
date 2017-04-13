#ifndef ONEFLOW_JOB_ID_MANAGER_H_
#define ONEFLOW_JOB_ID_MANAGER_H_

#include "common/util.h"
#include "job/resource.pb.h"

namespace oneflow {

// Phy  : Physical
// Loc  : Local
// Thrd : Thread

using DevicePhyId = int32_t;

using MachineId = int32_t;

using ThrdGloId = int32_t;
using ThrdLocId = int32_t;

class IDMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDMgr);
  ~IDMgr() = default;

  static IDMgr& Singleton() {
    static IDMgr obj;
    return obj;
  }

  void Init(const Resource& resource) {
    TODO();
  }

  ThrdLocId ThrdLocId4DevicePhyId(DevicePhyId) const {
    TODO();
  }
  ThrdLocId DiskThrdLocId() const {
    TODO();
  }
  ThrdLocId BoxingThrdLocId() const {
    TODO();
  }
  ThrdLocId CommNetThrdLocId() const {
    TODO();
  }

  int32_t NewNodeId();
  int32_t NewEdgeId();
  int32_t NewRegstDescId();

 private:
  IDMgr() = default;

};

} // namespace oneflow

#endif // ONEFLOW_JOB_ID_MANAGER_H_
