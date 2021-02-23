/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_

#define DEPRECATED __attribute__((deprecated))

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/common/platform.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/channel.h"

namespace oneflow {

struct CommNetItem {
  bool is_read;
  std::function<void()> callback;
  CommNetItem() : CommNetItem(false, nullptr) {}
  CommNetItem(bool read, const std::function<void()>& cb) : is_read(read), callback(cb) {}
};

class CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNet);
  virtual ~CommNet();

  // "RegisterMemory" will return a Token, after "RegisterMemoryDone",
  // we can use this token to use the "Read"
  virtual void* RegisterMemory(void* ptr, size_t byte_size) = 0;
  virtual void UnRegisterMemory(void* token) = 0;
  virtual void RegisterMemoryDone() = 0;

  // Stream
  void* NewActorReadId();
  void DeleteActorReadId(void* actor_read_id);
  void Read(void* actor_read_id, int64_t src_machine_id, void* src_token, void* dst_token);
  void AddReadCallBack(void* actor_read_id, std::function<void()> callback);
  void ReadDone(void* read_id);

  virtual void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) = 0;

 protected:
  CommNet();
  DEPRECATED CommNet(const Plan& plan);

  virtual void DoRead(void* read_id, int64_t src_machine_id, void* src_token, void* dst_token) = 0;
  const HashSet<int64_t>& peer_machine_id() { return peer_machine_id_; }

  Channel<std::function<void()>> ready_cbs_;

 private:
  friend class Global<CommNet>;
  void AddWorkToStream(void* actor_read_id, const std::function<void()>& cb, bool is_read);
  struct ActorReadContext;
  struct ReadContext {
    ActorReadContext* actor_read_ctx;
  };
  struct ActorReadContext {
    std::mutex waiting_list_mtx;
    std::list<CommNetItem> waiting_list;
  };
  HashSet<int64_t> peer_machine_id_;
  std::thread ready_cb_poller_;
};

template<typename MemDescType>
class CommNetIf : public CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNetIf);
  CommNetIf() = default;
  DEPRECATED CommNetIf(const Plan& plan) : CommNet(plan) {}
  virtual ~CommNetIf() {}

  void* RegisterMemory(void* ptr, size_t byte_size) override {
    MemDescType* mem_desc = NewMemDesc(ptr, byte_size);
    std::unique_lock<std::mutex> lck(mem_descs_mtx_);
    CHECK(mem_descs_.insert(mem_desc).second);
    return mem_desc;
  }

  void UnRegisterMemory(void* token) override {
    MemDescType* mem_desc = static_cast<MemDescType*>(token);
    delete mem_desc;
    std::unique_lock<std::mutex> lck(mem_descs_mtx_);
    CHECK_EQ(mem_descs_.erase(mem_desc), 1);
  }

 protected:
  virtual MemDescType* NewMemDesc(void* ptr, size_t byte_size) = 0;
  const HashSet<MemDescType*>& mem_descs() { return mem_descs_; }

 private:
  std::mutex mem_descs_mtx_;
  HashSet<MemDescType*> mem_descs_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
