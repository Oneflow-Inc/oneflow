#ifndef ONEFLOW_CORE_ACTOR_REGST_HANDLER_H_
#define ONEFLOW_CORE_ACTOR_REGST_HANDLER_H_

#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/thread/thread_context.h"
#include "oneflow/core/actor/register_slot.h"

namespace oneflow {

namespace actor {

struct MsgDeliveryCtx {
  int64_t actor_id;
  DeviceCtx* device_ctx;
  MsgDeliveryCtx(int64_t id, DeviceCtx* ctx) : actor_id(id), device_ctx(ctx) {}
};

struct ActorMsgUtil {
  static void AsyncSendMsg(MsgDeliveryCtx* msg_ctx, const ActorMsg& msg) {
    std::function<void()> callback = [msg]() { Global<ActorMsgBus>::Get()->SendMsg(msg); };
    if (Global<IDMgr>::Get()->GlobalWorkStreamId4ActorId(msg_ctx->actor_id)
        == Global<IDMgr>::Get()->GlobalWorkStreamId4ActorId(msg.dst_actor_id())) {
      callback();
    } else {
      msg_ctx->device_ctx->AddCallBack(callback);
    }
  }
};

using ProducedRegstType = HashMap<int64_t, std::vector<std::unique_ptr<Regst>>>;

class RegstHandlerIf {
 public:
  virtual void Init(const RegstHandlerProto&, const ProducedRegstType&, MsgDeliveryCtx*,
                    std::shared_ptr<void>) = 0;
  virtual std::string type() = 0;

  virtual Regst* GetRegstByRegstDescId(int64_t) const = 0;

  virtual void UpdateWithEordMsg(const ActorMsg&) = 0;
  virtual void UpdateWithRegstMsg(const ActorMsg&) = 0;
  virtual void UpdateWithProducedRegstMsg(const ActorMsg&) = 0;

  virtual bool IsReady() const = 0;
  virtual void HandleRegstMsgAfterAct() = 0;
  virtual bool NoLongerConsumeRegst() const = 0;
  virtual bool NoLongerConsumedByOthers() const = 0;
};

class NormalRegstHandler : public RegstHandlerIf {
 public:
  void Init(const RegstHandlerProto&, const ProducedRegstType&, MsgDeliveryCtx*,
            std::shared_ptr<void>) override final;
  bool NoLongerConsumeRegst() const override final {
    return (eord_cnt_ == consumed_rs_.total_regst_desc_cnt());
  }
  bool NoLongerConsumedByOthers() const override final { return total_reading_cnt_ == 0; }
  void UpdateWithRegstMsg(const ActorMsg&) override final;
  void UpdateWithEordMsg(const ActorMsg&) override final;
  bool IsReady() const override final;
  Regst* GetRegstByRegstDescId(int64_t) const override final;
  void HandleRegstMsgAfterAct() override final;

  void ForEachRegstDescId(std::function<void(int64_t)>) const;

 protected:
  NormalRegstHandler() = default;
  ~NormalRegstHandler() = default;

  MsgDeliveryCtx* msg_delivery_ctx() { return msg_delivery_ctx_.get(); }
  RegstSlot* mut_consumed_rs() { return &consumed_rs_; }
  RegstSlot* mut_produced_rs() { return &produced_rs_; }
  void* mut_kernel_other() { return kernel_other_.get(); }

  int64_t ReadingCnt4ProducedRegst(Regst* regst) const {
    return produced_regst2reading_cnt_.at(regst);
  }
  void UpdateReadingCnt4ProducedRegst(Regst* regst, int64_t update_val) {
    produced_regst2reading_cnt_.at(regst) += update_val;
    total_reading_cnt_ += update_val;
  }

 private:
  virtual void DerivedInit(const RegstHandlerProto&) {}
  virtual void UpdateWithConsumedRegstMsg(const ActorMsg&) = 0;
  virtual void HandleConsumedRegstAfterAct() = 0;
  virtual void HandleProducedRegstAfterAct() = 0;

  void InsertNewRegstDescId(bool is_produced, int64_t regst_desc_id) {
    if (is_produced) {
      produced_rs_.InsertRegstDescId(regst_desc_id);
    } else {
      consumed_rs_.InsertRegstDescId(regst_desc_id);
      consumed_regst2eord_.emplace(regst_desc_id, false);
    }
  }

  std::unique_ptr<MsgDeliveryCtx> msg_delivery_ctx_;
  std::shared_ptr<void> kernel_other_;

  RegstSlot consumed_rs_;
  RegstSlot produced_rs_;

  HashMap<int64_t, bool> consumed_regst2eord_;
  int64_t eord_cnt_;
  HashMap<Regst*, int64_t> produced_regst2reading_cnt_;
  int64_t total_reading_cnt_;
  // HashMap<int64_t, int64_t> produced_regst2expected_act_id_;
};

class CtrlRegstHandler final : public NormalRegstHandler {
 public:
  std::string type() override { return "Ctrl"; }
  void UpdateWithConsumedRegstMsg(const ActorMsg&) override;
  void UpdateWithProducedRegstMsg(const ActorMsg&) override;
  void HandleConsumedRegstAfterAct() override;
  void HandleProducedRegstAfterAct() override;
};

class NaiveRegstHandler final : public NormalRegstHandler {
 public:
  std::string type() override { return "Naive"; }
  void UpdateWithConsumedRegstMsg(const ActorMsg&) override;
  void UpdateWithProducedRegstMsg(const ActorMsg&) override;
  void HandleConsumedRegstAfterAct() override;
  void HandleProducedRegstAfterAct() override;
};

class InplaceRegstHandler final : public NormalRegstHandler {
 public:
  std::string type() override { return "Inplace"; }
  void DerivedInit(const RegstHandlerProto&) override;
  void UpdateWithConsumedRegstMsg(const ActorMsg&) override;
  void UpdateWithProducedRegstMsg(const ActorMsg&) override;
  void HandleConsumedRegstAfterAct() override;
  void HandleProducedRegstAfterAct() override;

 private:
  HashMap<int64_t, int64_t> inplace_pair_in2out_;
  HashMap<int64_t, int64_t> inplace_pair_out2in_;
};

}  // namespace actor

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REGST_HANDLER_H_
