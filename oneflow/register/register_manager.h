#ifndef ONEFLOW_REGISTER_REGISTER_MANAGER_H_
#define ONEFLOW_REGISTER_REGISTER_MANAGER_H_

#include "register/register.h"
#include "common/util.h"
#include "common/id_manager.h"
#include "common/ofelf.pb.h"
#include "memory/memory_manager.h"
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
  
  void NewRegstFromRegstDesc(
      uint64_t producer_id,
      const RegstDescProto& regstdesc,
      std::size_t sizeof_floating,
      HashMap<uint64_t, HashSet<uint64_t>> actor_id2produced_regst_desc_id,
      HashMap<uint64_t, std::vector<uint64_t>> regst_desc_id2regst_ids);

 private:
  RegstMgr();
  HashMap<uint64_t, std::unique_ptr<Regst>> regst_id2regst_;

};

}
#endif
