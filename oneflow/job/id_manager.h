#ifndef ONEFLOW_JOB_ID_MANAGER_H_
#define ONEFLOW_JOB_ID_MANAGER_H_

#include "common/util.h"
#include "job/resource.pb.h"

namespace oneflow {

class IDMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDMgr);
  ~IDMgr() = default;

  static IDMgr& Singleton() {
    static IDMgr obj;
    return obj;
  }

  void Init() {
    TODO();
  }

  int64_t ThrdLocId4DevicePhyId(int64_t) const { TODO(); }
  int64_t DiskThrdLocId() const { TODO(); }
  int64_t BoxingThrdLocId() const { TODO(); }
  int64_t CommNetThrdLocId() const { TODO(); }

  int64_t NewNodeId() const { TODO(); }
  int64_t NewEdgeId() const { TODO(); }
  int64_t NewRegstDescId() const { TODO(); }

 private:
  IDMgr() = default;

};

} // namespace oneflow

#endif // ONEFLOW_JOB_ID_MANAGER_H_
