#ifndef ONEFLOW_REGISTER_REGISTER_H_
#define ONEFLOW_REGISTER_REGISTER_H_

#include "register/blob.h"
#include "actor/actor_message.pb.h"
#include "actor/actor_msg_bus.h"
#include "common/util.h"

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
  mutable std::mutex mtx_;
  uint64_t id_;
  int32_t cnt_;
  uint64_t producer_id_;
  std::vector<uint64_t> consumer_ids_;
  std::function<void()> deleter_;
  HashMap<std::string, std::unique_ptr<Blob>> lbn2blob_;
};

} // namespace oneflow

#endif // ONEFLOW_REGISTER_REGISTER_H_
