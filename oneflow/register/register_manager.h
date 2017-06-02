#ifndef ONEFLOW_REGISTER_REGISTER_MANAGER_H_
#define ONEFLOW_REGISTER_REGISTER_MANAGER_H_

#include "oneflow/register/register.h"
#include "oneflow/common/util.h"
#include "oneflow/common/id_manager.h"
#include "oneflow/common/ofelf.pb.h"
#include "oneflow/memory/memory_allocator.h"
#include "oneflow/runtime/runtime_info.h"

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
