#ifndef ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_

#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/register/register.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

class RegstMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstMgr);
  RegstMgr() = delete;
  ~RegstMgr();

  void NewRegsts(const RegstDescProto& regst_desc_proto, DeviceType device_type,
                 std::function<void(Regst*)> OneRegstDone);

 private:
  friend class Global<RegstMgr>;
  RegstMgr(const Plan& plan);

  HashMap<int64_t, std::unique_ptr<const RtRegstDesc>> regst_desc_id2rt_regst_desc_;
  HashMap<int64_t, char*> regst_desc_id2mem_ptr_;
  std::list<std::function<void()>> deleters_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_
