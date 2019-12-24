#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutUserOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutUserOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override {
    return !op_conf().user_conf().input().empty();
    // return op_conf().record_load_conf().has_tick();
  }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    (*ret.mutable_user_conf()->mutable_input())["in"].add_s(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kUserConf, MutUserOpConTickInputHelper);

}  // namespace oneflow
