#ifndef ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
#define ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_

#include <mutex>
#include "oneflow/core/common/thread_safe_counter.h"
#include "oneflow/core/distributed_runtime/worker.pb.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/network/network_memory.h"
#include "oneflow/core/persistence/persistent_circular_line_reader.h"

namespace oneflow {
// Only for pairs of std::hash-able types for simplicity.
// You can of course template this struct to allow other hash functions
struct pair_hash {
  template<class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);

    // Mainly for demonstration purposes, i.e. works but is overly simple
    // In the real world, use sth. like boost.hash_combine
    return h1 ^ h2;
  }
};

struct NetMemoryDescriptor {
  void* regst_ptr;
  void* local_ptr;
  void* network_ptr;
  int64_t this_machine_id;
  std::vector<int64_t> consumer_machine_ids;
  std::vector<int64_t> consumer_task_ids;
};

class RuntimeCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeCtx);
  ~RuntimeCtx() = default;

  OF_SINGLETON(RuntimeCtx);

  int64_t this_machine_id() const { return this_machine_id_; }

  void set_this_machine_name(const std::string& name);

  ThreadSafeCounter& mut_model_init_cnt() { return model_init_cnt_; }

  PersistentCircularLineReader* GetDataReader() { return data_reader_.get(); }
  void InitDataReader(const std::string& filepath);

  ThreadSafeCounter& mut_active_actor_cnt() { return active_actor_cnt_; }
  ThreadSafeCounter& mut_inactive_actor_cnt() { return inactive_actor_cnt_; }

  void AddLocalNetMemoryDesc(const NetMemoryDescriptor& net_memory_desc);
  const std::vector<NetMemoryDescriptor>& local_net_memory_descs() const;

  void AddRemoteMemoryDescriptor(int64_t machine_id,
                                 const RemoteRegstDesc& remote_regst_desc);
  const MemoryDescriptor& memory_descriptor(int64_t consumer_task_id,
                                            uint64_t regst_address) const;

 private:
  RuntimeCtx() { LOG(INFO) << "RuntimeCtx Init"; }

  int64_t this_machine_id_;
  std::string this_machine_name_;

  ThreadSafeCounter model_init_cnt_;

  std::unique_ptr<PersistentCircularLineReader> data_reader_;

  ThreadSafeCounter active_actor_cnt_;
  ThreadSafeCounter inactive_actor_cnt_;

  mutable std::mutex mutex_;
  std::vector<NetMemoryDescriptor> local_net_memory_descs_;
  // <consumer_task_id, regst_address>
  std::unordered_map<std::pair<int64_t, uint64_t>, MemoryDescriptor, pair_hash>
      remote_net_memory_descs_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
