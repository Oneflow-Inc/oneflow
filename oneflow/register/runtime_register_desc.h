#ifndef ONEFLOW_REGISTER_RUNTIME_REGISTER_DESC_H_
#define ONEFLOW_REGISTER_RUNTIME_REGISTER_DESC_H_

#include "oneflow/common/util.h"
#include "oneflow/memory/memory_case.pb.h"
#include "oneflow/register/register_desc.pb.h"

namespace oneflow {

class RtRegstDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RtRegstDesc);
  RtRegstDesc() = delete;
  ~RtRegstDesc() = default;

  RtRegstDesc(const RegstDescProto&) { TODO(); }

 private:
  uint64_t regst_desc_id_;
  uint64_t producer_actor_id_;
  std::vector<uint64_t> subscribers_actor_id_;
  std::unordered_map<std::string, std::unique_ptr<Shape>> lbn2shape_;
  int64_t register_num_;
  MemoryCase mem_case_;

};

} // namespace oneflow

#endif // ONEFLOW_REGISTER_RUNTIME_REGISTER_DESC_H_
