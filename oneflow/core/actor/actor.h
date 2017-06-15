#ifndef ONEFLOW_CORE_ACTOR_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_H_

#include <queue>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/cuda_stream_handle.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/actor/cpu_device_context.h"
#include "oneflow/core/actor/cuda_device_context.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/register/register.h"
#include "oneflow/core/register/local_register_warpper.h"
#include "oneflow/core/register/remote_register_warpper.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/thread/thread_context.h"

namespace oneflow {

class Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Actor);
  virtual ~Actor() = default;

  virtual void Init(const TaskProto& task_proto) = 0;
  // 1: success, and actor finish
  // 0: success, and actor not finish
  virtual int ProcessMsg(const ActorMsg&, const ThreadContext& ctx) = 0;

  uint64_t actor_id() const { return actor_id_; }
 
 protected:
  struct ExecKernel {
    const Kernel* kernel;
    HashMap<std::string, uint64_t> bn_in_op2regst_desc_id;
  };

  Actor() = default;
  uint64_t RegstDescId4Name(const std::string& name) const {
    return name2regst_desc_id_.at(name);
  }

  std::unique_ptr<DeviceCtx>& mut_device_ctx() { return device_ctx_; }
  KernelCtx GenDefaultKernelCtx() const;

  // Status of Produced Registers
  uint64_t expected_piece_id() const { return expected_piece_id_; }
  void AsyncWardKernel(
      const KernelCtx&,
      std::function<std::shared_ptr<RegstWarpper>(uint64_t)> Regst4RegstDescId);
  void AsyncSendReadableRegstMsg();
  void AsyncSendRegstDescDoneMsgToSubscribers(uint64_t regst_desc_id);
  void AsyncDo(std::function<void()>);
  int TryUpdtStateAsProducedRegst(Regst* regst);
  Regst* GetCurWriteableRegst(uint64_t regst_desc_id);
  Regst* GetCurWriteableRegst(const std::string& name);
  void ForEachCurWriteableRegst(std::function<void(Regst*)> func);
  bool IsWriteReady();
  void SetReadOnlyForRegstDescId(uint64_t regst_desc_id) {
    auto it = writeable_produced_regst_.find(regst_desc_id);
    if (!it->second.empty()) { writeable_produced_regst_desc_num_ -= 1; }
    writeable_produced_regst_.erase(it);
  }
  int64_t total_reading_cnt() const { return total_reading_cnt_; }

 private:
  uint64_t actor_id_;
  KernelWardFunc ward_func_;
  std::vector<ExecKernel> exec_kernel_vec_;
  HashMap<uint64_t, std::vector<std::unique_ptr<Regst>>> produced_regsts_; // <regst_desc_id, regst>
  HashMap<std::string, uint64_t> name2regst_desc_id_;

  std::unique_ptr<DeviceCtx> device_ctx_;
  
  // Status of Produced Registers
  uint64_t expected_piece_id_;
  HashMap<uint64_t, std::queue<Regst*>> writeable_produced_regst_; // <regst_desc_id, regst>
  uint64_t writeable_produced_regst_desc_num_;
  HashMap<Regst*, int64_t> produced_regst2reading_cnt_;
  int64_t total_reading_cnt_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_ACTOR_H_
