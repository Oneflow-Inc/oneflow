#ifndef ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_
#define ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/register/register_desc.pb.h"

namespace oneflow {

class RtRegstDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RtRegstDesc);
  RtRegstDesc() = delete;
  ~RtRegstDesc() = default;

  RtRegstDesc(const RegstDescProto& regst_desc_proto);

  uint64_t regst_desc_id() const { return regst_desc_id_; }
  uint64_t producer_actor_id() const { return producer_actor_id_; }
  const std::vector<uint64_t>& subscribers_actor_id() const { 
    return subscribers_actor_id_;
  }
  int64_t register_num() const { return register_num_; }
  const MemoryCase& mem_case() const { return mem_case_; }

  const Shape* GetShapePtrFromLbn(const std::string& lbn) const {
    return lbn2shape_.at(lbn).get();
  }

 private:
  uint64_t regst_desc_id_;
  uint64_t producer_actor_id_;
  std::vector<uint64_t> subscribers_actor_id_;
  std::unordered_map<std::string, std::unique_ptr<Shape>> lbn2shape_;
  int64_t register_num_;
  MemoryCase mem_case_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_REGISTER_RUNTIME_REGISTER_DESC_H_
