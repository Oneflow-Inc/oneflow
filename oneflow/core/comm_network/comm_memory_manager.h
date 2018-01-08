#ifndef ONEFLOW_CORE_COMM_NETWORK_COMM_MEMORY_MANAGER_H_
#define ONEFLOW_CORE_COMM_NETWORK_COMM_MEMORY_MANAGER_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename MemType>
class CommMemMgr {
 public:
  CommMemMgr() : unregister_comm_mems_cnt_(0) { comm_mems_.clear(); }
  ~CommMemMgr() { CHECK_EQ(unregister_comm_mems_cnt_, 0); }

  void RegisterCommMem(MemType *comm_mem) {
    std::unique_lock<std::mutex> lck(comm_mem_mtx_);
    comm_mems_.push_back(comm_mem);
  }

  void UnRegisterCommMem() {
    std::unique_lock<std::mutex> lck(comm_mem_mtx_);
    CHECK(!comm_mems_.empty());
    unregister_comm_mems_cnt_ += 1;
    if (unregister_comm_mems_cnt_ == comm_mems_.size()) {
      for (MemType *comm_mem : comm_mems_) { delete comm_mem; }
      comm_mems_.clear();
      unregister_comm_mems_cnt_ = 0;
    }
  }

  const std::list<MemType *> &comm_mems() { return comm_mems; }

 private:
  size_t unregister_comm_mems_cnt_;
  std::list<MemType *> comm_mems_;
  std::mutex comm_mem_mtx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_COMM_MEMORY_MANAGER_H
