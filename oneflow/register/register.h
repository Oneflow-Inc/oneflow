#ifndef ONEFLOW_REGISTER_REGISTER_H_
#define ONEFLOW_REGISTER_REGISTER_H_

#include "register/blob.h"
#include "job/message.h"
#include "thred/commbus.h"
#include "common/util.h"

namespace oneflow {

class Regst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Regst);
  ~Regst() = default;
  
  void ProduceDone();
  void ConsumeDone();

  Blob* GetBlobFromLbn(const std::string& lbn);

 private:
  friend class RegstMgr;
  Regst();
  uint64_t id_;
  std::atomic<int32_t> cnt_;
  uint64_t producer_id_;
  std::vector<uint64_t> consumer_ids_;
  HashMap<std::string, std::unique_ptr<Blob>> lbn2blob_;
};

}
#endif
