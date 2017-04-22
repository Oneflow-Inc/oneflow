#ifndef ONEFLOW_JOB_ID_MANAGER_H_
#define ONEFLOW_JOB_ID_MANAGER_H_

#include "common/util.h"
#include "job/resource.pb.h"
#include "job/id_manager.pb.h"

namespace oneflow {

class IDMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDMgr);
  ~IDMgr() = default;

  static IDMgr& Singleton() {
    static IDMgr obj;
    return obj;
  }

  void InitFromResource(const Resource&) {
    TODO();
  }

  // Compile
  uint64_t MachineID4MachineName(const std::string& machine) const { TODO(); }
  uint64_t ThrdLocId4DevicePhyId(uint64_t) const { TODO(); }
  uint64_t DiskThrdLocId() const { TODO(); }
  uint64_t BoxingThrdLocId() const { TODO(); }
  uint64_t CommNetThrdLocId() const { TODO(); }

  uint64_t NewTaskId(uint64_t machine_id, uint64_t thrd_local_id) const { TODO(); }
  uint64_t NewRegstDescId(uint64_t producer_task_id) const { TODO(); }

  // Runtime

 private:
  IDMgr() = default;

};

} // namespace oneflow

#endif // ONEFLOW_JOB_ID_MANAGER_H_
