#ifndef ONEFLOW_CORE_FRAMEWORK_RPC_TOKEN_H_
#define ONEFLOW_CORE_FRAMEWORK_RPC_TOKEN_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/type_traits.h"

namespace oneflow {

class ParallelDesc;

enum RpcTokenCmdLocalMajor {
	// Begin
	kCheckingParallelConfSizeCmdLocalMajor = 0,
	kInitializingPlacementCmdLocalMajor,
	kDataSendRecvCmdLocalMajor,
	// End 
	kRpcTokenCmdLocalMajorSize,
};

class RpcToken final {
 public: 
  RpcToken(uint32_t major, uint32_t minor) : major_(major), minor_(minor) {}
  RpcToken(const RpcToken&) = default;
  RpcToken(RpcToken&) = default;
  ~RpcToken(RpcToken&) = default;

	static const uint32_t kStartTokenMajor4Cmd = kRpcTokenCmdLocalMajorSize;
	static const uint32_t kStartTokenMajor4Placement = 4096;

  uint32_t major() const { return major_; }
  uint32_t minor() const { return minor_; }
  operator uint64_t() const { return static_cast<uint64_t>(major_) << 32 + minor_; }

  RpcToken& operator++() {
    ++minor_;
    return *this;
  }

 private:
  uint32_t major_;
  uint32_t minor_;
};

static_assert(sizeof(RpcToken) == sizeof(uint64_t), "");

template<>
struct IsScalarType<RpcToken> final {
  static const bool value = true;
};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_RPC_TOKEN_H_
