#ifndef ONEFLOW_REGISTER_RUNTIME_REGISTER_DESC_H_
#define ONEFLOW_REGISTER_RUNTIME_REGISTER_DESC_H_

#include "common/util.h"
#include "memory/memory_case.pb.h"
#include "register/register_desc.pb.h"

namespace oneflow {

class RtRegstDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RtRegstDesc);
  RtRegstDesc() = delete;
  ~RtRegstDesc() = default;

  RtRegstDesc(const RegstDescProto&) { TODO(); }

  // TODO: Add Getter

 private:
  uint64_t regst_desc_id_;
  uint64_t producer_task_id_;
  std::vector<uint64_t> subscribers_task_id_;
  std::unordered_map<std::string, std::unique_ptr<Shape>> lbn2shape_;
  int64_t register_num_;
  MemoryCase mem_case_;

};

} // namespace oneflow

#endif // ONEFLOW_REGISTER_RUNTIME_REGISTER_DESC_H_
