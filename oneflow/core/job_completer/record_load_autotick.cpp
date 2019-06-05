#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutRecordLoadOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutRecordLoadOpConTickInputHelper() : MutOpConTickInputHelper() {}

  bool VirtualIsTickInputBound() const override { return op_conf().record_load_conf().has_tick(); }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_record_load_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kRecordLoadConf, MutRecordLoadOpConTickInputHelper);

}  // namespace oneflow
