#ifndef ONEFLOW_CORE_COMM_NETWORK_MEMORY_DESC_MANAGER_H_
#define ONEFLOW_CORE_COMM_NETWORK_MEMORY_DESC_MANAGER_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename MemDescType>
class MemDescMgr final {
 public:
  MemDescMgr() : unregister_mem_descs_cnt_(0) { mem_descs_.clear(); }
  ~MemDescMgr() { CHECK_EQ(unregister_mem_descs_cnt_, 0); }

  void RegisterMemDesc(MemDescType* mem_desc) {
    std::unique_lock<std::mutex> lck(mem_desc_mtx_);
    mem_descs_.push_back(mem_desc);
  }

  void UnRegisterMemDesc() {
    std::unique_lock<std::mutex> lck(mem_desc_mtx_);
    CHECK(!mem_descs_.empty());
    unregister_mem_descs_cnt_ += 1;
    if (unregister_mem_descs_cnt_ == mem_descs_.size()) {
      for (MemDescType* mem_desc : mem_descs_) { delete mem_desc; }
      mem_descs_.clear();
      unregister_mem_descs_cnt_ = 0;
    }
  }

  const std::list<MemDescType*>& mem_descs() { return mem_descs_; }

 private:
  size_t unregister_mem_descs_cnt_;
  std::list<MemDescType*> mem_descs_;
  std::mutex mem_desc_mtx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_MEMORY_DESC_MANAGER_H
