#include "oneflow/core/framework/rpc_token.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

RpcToken::operator uint64_t() const {
  static_assert(sizeof(RpcToken) == sizeof(uint64_t), "");
  return *reinterpret_cast<const uint64_t*>(this);
}

RpcToken& RpcToken::operator++() {
  seq_id_ = (seq_id_ + 1) % (1 << 24);
  return *this;
}

}
