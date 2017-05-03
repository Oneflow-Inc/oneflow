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
    return of_make_unique<RegstDesc> ();
  }

 private:
  RegstDescMgr() = default;
};

} // namespace oneflow

#endif // ONEFLOW_REGISTER_REGISTER_DESC_MANAGER_H_
