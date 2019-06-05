#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutDecodeRandomOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutDecodeRandomOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override {
    return op_conf().decode_random_conf().has_tick();
  }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_decode_random_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kDecodeRandomConf, MutDecodeRandomOpConTickInputHelper);

}  // namespace oneflow
