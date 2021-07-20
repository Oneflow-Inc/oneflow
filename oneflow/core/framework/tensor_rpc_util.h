#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_RPC_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_RPC_UTIL_H_

#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/framework/tensor.h"

namespace oneflow {

class FlatConsistentTensorMeta;

class CheckConsistencyAsyncRpcCtx : public AsyncRpcCtx {
 public:
	CheckConsistencyAsyncRpcCtx(
		const std::shared_ptr<const Shape>& shape, DataType dtype, const RpcToken& rpc_token,
		Symbol<ParallelDesc> parallel_desc, Symbol<cfg::ParallelDistribution> parallel_distribution)
		: shape_(shape), dtype_(dtype), rpc_token_(rpc_token),
			parallel_desc_(parallel_desc), parallel_distribution_(parallel_distribution) {}

	~CheckConsistencyAsyncRpcCtx() override;

	Maybe<void> MakeDataBufferAndCallback(
		int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) override;

	Maybe<void> Check();

 private:
	std::shared_ptr<const Shape> shape_;
	DataType dtype_;
	RpcToken rpc_token_;
	Symbol<ParallelDesc> parallel_desc_;
	Symbol<cfg::ParallelDistribution> parallel_distribution_;

	std::shared_ptr<FlatConsistentTensorMeta> flatten_consistent_tensor_meta_;
};

Maybe<CheckConsistencyAsyncRpcCtx> LaunchTensorMetaConsistencyCheck(const one::Tensor& tensor);

}

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_RPC_UTIL_H_
