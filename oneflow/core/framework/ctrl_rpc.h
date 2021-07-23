#ifndef ONEFLOW_CORE_FRAMEWORK_CTRL_RPC_H_
#define ONEFLOW_CORE_FRAMEWORK_CTRL_RPC_H_

#include "oneflow/core/framework/rpc_token.h"
#include "oneflow/core/framework/rpc_util.h"

namespace oneflow {

class Shape;
class FlatShape;

struct CtrlRpc {

  // Returns rank_to_flat_shape.
	static Maybe<Hash<int64_t, std::shared_ptr<FlatShape>>> All2AllSyncShape(const Shape&);

};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_CTRL_RPC_H_
