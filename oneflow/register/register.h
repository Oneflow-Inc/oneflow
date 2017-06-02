#ifndef ONEFLOW_REGISTER_REGISTER_H_
#define ONEFLOW_REGISTER_REGISTER_H_

#include "oneflow/register/blob.h"
#include "oneflow/common/util.h"
#include "oneflow/register/runtime_register_desc.h"

namespace oneflow {

class Regst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Regst);
  ~Regst() {
    deleter_();
  }
  
  Blob* GetBlobPtrFromLbn(const std::string& lbn);

  uint64_t regst_desc_id() const {
    TODO();
  }
  uint64_t producer_actor_id() const {
    TODO();
  }
  const std::vector<uint64_t>& subscribers_actor_id() const {
    TODO();
  }
    
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
