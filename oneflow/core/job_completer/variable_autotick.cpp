#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutVariableOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutVariableOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override { return op_conf().variable_conf().has_tick(); }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_variable_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kVariableConf, MutVariableOpConTickInputHelper);

}  // namespace oneflow
