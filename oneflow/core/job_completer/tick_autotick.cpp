#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutTickOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutTickOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override { return op_conf().tick_conf().has_tick(); }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_tick_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kTickConf, MutTickOpConTickInputHelper);

}  // namespace oneflow
