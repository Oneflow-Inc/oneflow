#ifndef ONEFLOW_REGISTER_REGISTER_DESC_MANAGER_H_
#define ONEFLOW_REGISTER_REGISTER_DESC_MANAGER_H_

#include "register/register_desc.h"

namespace oneflow {

class RegstDescMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstDescMgr);
  RegstDescMgr() = default;
  ~RegstDescMgr() = default;

  std::unique_ptr<RegstDesc> CreateRegisterDesc() {
    return of_make_unique<RegstDesc> ();
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_REGISTER_REGISTER_DESC_MANAGER_H_
