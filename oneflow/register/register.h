#ifndef ONEFLOW_REGISTER_REGISTER_H_
#define ONEFLOW_REGISTER_REGISTER_H_

#include "register/blob.h"
#include "common/util.h"
#include "register/runtime_register_desc.h"

namespace oneflow {

class Regst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Regst);
  ~Regst() {
    deleter_();
  }
  
  Blob* GetBlobPtrFromLbn(const std::string& lbn);
    
 private:
  friend class RegstMgr;
  Regst() = default;

  std::shared_ptr<const RtRegstDesc> regst_desc_;
  uint64_t regst_id_;
  std::function<void()> deleter_;
  HashMap<std::string, std::unique_ptr<Blob>> lbn2blob_;
};

} // namespace oneflow

#endif // ONEFLOW_REGISTER_REGISTER_H_
