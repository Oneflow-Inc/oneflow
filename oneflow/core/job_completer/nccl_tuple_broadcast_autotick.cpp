#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutNcclTupleBroadcastOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutNcclTupleBroadcastOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override {
    return op_conf().nccl_tuple_broadcast_conf().has_tick();
  }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_nccl_tuple_broadcast_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kNcclTupleBroadcastConf,
                   MutNcclTupleBroadcastOpConTickInputHelper);

}  // namespace oneflow
