#ifndef ONEFLOW_REGISTER_RUNTIME_REGISTER_DESC_H_
#define ONEFLOW_REGISTER_RUNTIME_REGISTER_DESC_H_

#include "common/util.h"
#include "common/protobuf.h"
#include "common/shape.h"
#include "memory/memory_case.pb.h"
#include "register/register_desc.pb.h"

namespace oneflow {

class RtRegstDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RtRegstDesc);
  RtRegstDesc() = delete;
  ~RtRegstDesc() = default;

  RtRegstDesc(const RegstDescProto& regst_desc_proto) {
    regst_desc_id_ = regst_desc_proto.regst_desc_id();
    producer_task_id_ = regst_desc_proto.producer_task_id();
    register_num_ = regst_desc_proto.register_num();

    const auto& subscriber = regst_desc_proto.subscriber_task_id();
    subscribers_task_id_ = std::vector<uint64_t>(subscriber.begin(), subscriber.end());

    for (const auto& pair : regst_desc_proto.lbn2shape()) {
      lbn2shape_.emplace(pair.first, of_make_unique<Shape>(pair.second));
    }
    mem_case_ = regst_desc_proto.mem_case();
  }

  uint64_t regst_desc_id() const { return regst_desc_id_; }
  uint64_t producer_task_id() const { return producer_task_id_; }
  std::vector<uint64_t>& subscribers_task_id() { 
    return subscribers_task_id_;
  }
  int64_t register_num() const { return register_num_; }
  MemoryCase mem_case() const { return mem_case_; }

  Shape* GetShapePtrFromLbn(const std::string& lbn) {
    return lbn2shape_.at(lbn).get();
  }

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
