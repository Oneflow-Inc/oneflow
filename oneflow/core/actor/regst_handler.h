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
  static void AsyncSendMsg(MsgDeliveryCtx*, const ActorMsg&);
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
  virtual void SendEordMsgForProducedRegst() = 0;
};

using RegstHandlerCreateFn = std::function<RegstHandlerIf*()>;

struct RegstHandlerRegistrar final {
  RegstHandlerRegistrar(const std::string&, RegstHandlerCreateFn);
};

RegstHandlerIf* CreateRegstHandler(const std::string&);

}  // namespace actor

#define REGISTER_REGST_HANDLER(handler_type, Handler)                                       \
  namespace {                                                                               \
  static actor::RegstHandlerRegistrar OF_PP_CAT(g_registrar, __LINE__)(handler_type, []() { \
    return new Handler();                                                                   \
  });                                                                                       \
  }  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REGST_HANDLER_H_
