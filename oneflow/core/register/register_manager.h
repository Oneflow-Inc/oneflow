#ifndef ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_

#include "oneflow/core/register/register.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/id_manager.h"
#include "oneflow/core/common/ofelf.pb.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/runtime/runtime_info.h"

namespace oneflow {

class RegstMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstMgr);
  ~RegstMgr() = default;

  static RegstMgr& Singleton() {
    static RegstMgr obj;
    return obj;
  }

  void NewRegsts(const RegstDescProto& regst_desc_proto,
                 std::function<void(Regst*)> OneRegstDone);
  
 private:
  RegstMgr() = default;
  
};

} // namespace oneflow

#endif // ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_
