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

  Regst* GetRegstFromRegstID(uint64_t regst_id) {
    return regst_id2regst_.at(regst_id).get();
  }
  
  void InitFromProto(const OfElf& ofelf);
  
 private:
  RegstMgr() = default;
  void NewRegstFromRegstDesc(
      uint64_t producer_id,
      uint64_t device_id,
      const std::pair<int, bool>& producer_type,
      const RegstDescProto& regstdesc,
      std::size_t sizeof_floating,
      const std::vector<uint64_t>& subscriber_ids,
      const HashSet<std::pair<int, bool>, pair_hash>& subscriber_types);
  
  HashMap<uint64_t, std::unique_ptr<Regst>> regst_id2regst_;
};

} // namespace oneflow

#endif // ONEFLOW_REGISTER_REGISTER_MANAGER_H_
