#ifndef ONEFLOW_REGISTER_REGISTER_MANAGER_H_
#define ONEFLOW_REGISTER_REGISTER_MANAGER_H_

#include "register/register_desc.pb.h"
#include "register/register.h"
#include "common/util.h"
#include "job/id_manager.h"
#include "job/ofelf.pb.h"
#include "task/task.pb.h"
#include "graph/exec_sequence.pb.h"

namespace oneflow {

class RegstMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstMgr);
  RegstMgr();
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
  HashMap<uint64_t, std::unique_ptr<Regst>> regst_id2regst_;

};

}
#endif
