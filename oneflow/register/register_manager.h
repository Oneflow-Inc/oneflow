#ifndef ONEFLOW_REGISTER_REGISTER_MANAGER_H_
#define ONEFLOW_REGISTER_REGISTER_MANAGER_H_

#include "register/register.h"
#include "common/util.h"
#include "common/id_manager.h"
#include "common/ofelf.pb.h"
#include "memory/memory_allocator.h"
#include "runtime/runtime_info.h"

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

#endif // ONEFLOW_REGISTER_REGISTER_MANAGER_H_
