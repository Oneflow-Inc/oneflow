#ifndef ONEFLOW_REGISTER_REGISTER_DESC_MANAGER_H_
#define ONEFLOW_REGISTER_REGISTER_DESC_MANAGER_H_

#include "register/register_desc.h"

namespace oneflow {

class RegstDescMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstDescMgr);
  ~RegstDescMgr() = default;

  static RegstDescMgr& Singleton() {
    static RegstDescMgr obj;
    return obj;
  }

  std::unique_ptr<RegstDesc> CreateRegisterDesc() {
    auto ret = of_make_unique<RegstDesc> ();
    regst_descs_.push_back(ret.get());
    return ret;
  }

  void AllRegstsToProto(PbRpf<RegstDescProto>* ret) {
    ret->Clear();
    for (RegstDesc* regst : regst_descs_) {
      regst->ToProto(ret->Add());
    }
  }

 private:
  RegstDescMgr() = default;
  std::list<RegstDesc*> regst_descs_;

};

} // namespace oneflow

#endif // ONEFLOW_REGISTER_REGISTER_DESC_MANAGER_H_
