#include "oneflow/core/job_completer/autotick.h"

namespace oneflow {

namespace {

class MutDecodeOneRecOpConTickInputHelper final : public MutOpConTickInputHelper {
 public:
  MutDecodeOneRecOpConTickInputHelper() : MutOpConTickInputHelper() {}
  ~MutDecodeOneRecOpConTickInputHelper() = default;

  bool VirtualIsTickInputBound() const override {
    return op_conf().decode_onerec_conf().has_tick();
  }

  OperatorConf NewTickInputBoundOpConf(const std::string& lbn) const override {
    OperatorConf ret(op_conf());
    ret.mutable_decode_onerec_conf()->set_tick(lbn);
    return ret;
  }
};

}  // namespace

REGISTER_AUTO_TICK(OperatorConf::kDecodeOnerecConf, MutDecodeOneRecOpConTickInputHelper);

}  // namespace oneflow
