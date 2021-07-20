#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/object_msg/flag_msg.h"
#include "oneflow/core/common/shape_vec.h"

namespace oneflow {


// clang-format off

namespace {

FLAG_MSG_BEGIN(FlatShape);
  OF_PUBLIC Maybe<void> Init(const std::shared_ptr<const Shape>& shape) {
		CHECK_LE_OR_RETURN(shape->NumAxes(), SHAPE_MAX_AXIS_SIZE);
		this->set_num_axes(shape->NumAxes());
		for (int i = 0; i < this->num_axes(); ++i) { *this->mutable_dim()->Mutable(i) = shape->At(i); }
		return Maybe<void>::Ok();
	}
  OF_PUBLIC Maybe<void> Check(const std::shared_ptr<const Shape>& shape) {
		CHECK_EQ_OR_RETURN(this->num_axes(), shape->NumAxes());
		for (int i = 0; i < this->num_axes(); ++i) {
			CHECK_EQ_OR_RETURN(this->dim().Get(i), shape->At(i));
		}
		return Maybe<void>::Ok();
	}
	FLAT_MSG_DEFINE_OPTIONAL(int64_t, num_axes);
	FLAT_MSG_DEFINE_REPEATED(int64_t, dim, SHAPE_MAX_AXIS_SIZE);
FLAG_MSG_END(FlatShape);

FLAG_MSG_BEGIN(FlatSplitParallel);
	FLAT_MSG_DEFINE_OPTIONAL(int64_t, axis);
FLAG_MSG_END(FlatSplitParallel);

FLAG_MSG_BEGIN(FlatBroadcastParallel);
FLAG_MSG_END(FlatBroadcastParallel);

FLAG_MSG_BEGIN(FlatPartialSumParallel);
FLAG_MSG_END(FlatPartialSumParallel);

FLAG_MSG_BEGIN(FlatSbpParallel);
	Maybe<void> Init(const cfg::SbpParallel& sbp_parallel) {
		if (sbp_parallel.has_split_parallel()) {
			this->mutable_split_parallel()->set_axis(sbp_parallel.split_parallel().axis());
		} else if (sbp_parallel.has_broadcast_parallel()) {
			this->mutable_broadcast_parallel();
		} else if (sbp_parallel.has_partial_sum_parallel()) {
			this->mutable_partial_sum_parallel();
		} else {
			OF_UNIMPLEMENTED();
		}
		return Maybe<void>::Ok();
	}

	Maybe<void> Check(const cfg::SbpParallel& sbp_parallel) {
		if (sbp_parallel.has_split_parallel()) {
			CHECK_EQ_OR_RETURN(this->split_parallel().axis(), sbp_parallel.split_parallel().axis());
		} else if (sbp_parallel.has_broadcast_parallel()) {
			CHECK_OR_RETURN(this->has_broadcast_parallel());
		} else if (sbp_parallel.has_partial_sum_parallel()) {
			CHECK_OR_RETURN(this->has_partial_sum_parallel());
		} else {
			OF_UNIMPLEMENTED();
		}
		return Maybe<void>::Ok();
	}
	
  FLAT_MSG_DEFINE_ONEOF(parallel_type,
      FLAT_MSG_ONEOF_FIELD(FlatSplitParallel, split_parallel)
      FLAT_MSG_ONEOF_FIELD(FlatBroadcastParallel, broadcast_parallel)
      FLAT_MSG_ONEOF_FIELD(FlatPartialSumParallel, partial_sum_parallel));
FLAG_MSG_END(FlatSbpParallel);

FLAG_MSG_BEGIN(FlatParallelDistribution);
  OF_PUBLIC Maybe<void> Init(Symbol<cfg::ParallelDistribution> parallel_distribution) {
		this->set_size(parallel_distribution->sbp_parallel_size());
		for (int i = 0; i < this->size(); ++i) {
			JUST(this->mutable_sbp_parallel()->Init(parallel_distribution.sbp_parallel(i)));
		}
		return Maybe<void>::Ok();
	}

  OF_PUBLIC Maybe<void> Check(Symbol<cfg::ParallelDistribution> parallel_distribution) {
		CHECK_EQ_OR_RETURN(this->size(), parallel_distribution->sbp_parallel_size());
		for (int i = 0; i < this->size(); ++i) {
			JUST(this->sbp_parallel().Get(i).Check(parallel_distribution.sbp_parallel(i)));
		}
		return Maybe<void>::Ok();
	}

	FLAT_MSG_DEFINE_OPTIONAL(size_t, size);
	FLAT_MSG_DEFINE_REPEATED(FlatSbpParallel, sbp_parallel, SHAPE_MAX_AXIS_SIZE);
FLAG_MSG_END(FlatParallelDistribution);

}

FLAG_MSG_BEGIN(FlatConsistentTensorMeta);
  OF_PUBLIC Maybe<void> Init(
			const std::shared_ptr<const Shape>& shape, DataType dtype, const RpcToken& rpc_token,
			Symbol<cfg::ParallelDistribution> parallel_distribution) {
		clear();
		JUST(this->mutable_shape()->Init(shape));
		this->set_dtype(dtype);
		this->set_rpc_token(static_cast<uint64_t>(rpc_token));
		JUST(this->mutable_parallel_distribution->Init(parallel_distribution));
		return Maybe<void>::Ok();
	}
  OF_PUBLIC Maybe<void> Check(
			const std::shared_ptr<const Shape>& shape, DataType dtype, const RpcToken& rpc_token,
			Symbol<cfg::ParallelDistribution> parallel_distribution) {
		OF_RETURN_IF_ERROR(this->shape().Check(shape));
		CHECK_OR_RETURN(this->dtype() == dtype);
		CHECK_OR_RETURN(this->rpc_token() == static_cast<uint64_t>(rpc_token));
		OF_RETURN_IF_ERROR(this->parallel_distribution().Check(parallel_distribution));
		return Maybe<void>::Ok();
	}
	FLAT_MSG_DEFINE_OPTIONAL(FlatShape, shape);
	FLAT_MSG_DEFINE_OPTIONAL(DataType, dtype);
	FLAT_MSG_DEFINE_OPTIONAL(uint64_t, rpc_token);
	FLAT_MSG_DEFINE_OPTIONAL(FlatParallelDistribution, parallel_distribution);
FLAG_MSG_END(FlatConsistentTensorMeta);

// clang-format on

CheckConsistencyAsyncRpcCtx::~CheckConsistencyAsyncRpcCtx() {}

Maybe<void> CheckConsistencyAsyncRpcCtx::MakeDataBufferAndCallback(
		int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) {
	const auto& flat_consistent_meta = std::make_shared<FlatConsistentTensorMeta>();	
	*buffer = flat_consistent_meta.get();
	*size = sizeof(FlatConsistentTensorMeta);
	*Callback = [flat_consistent_meta]() {};
	return Maybe<void>::Ok();
}

Maybe<void> CheckConsistencyAsyncRpcCtx::Check() {
	JUST(flatten_consistent_tensor_meta_->Check(
				shape_, dtype_, rpc_token_, parallel_distribution_));
	return Maybe<void>::Ok();
}

namespace {

class SendConsistencyAsyncRpcCtx : public AsyncRpcCtx {
 public:
	SendConsistencyAsyncRpcCtx(
		const std::shared_ptr<const Shape>& shape, DataType dtype, const RpcToken& rpc_token,
		Symbol<ParallelDesc> parallel_desc, Symbol<cfg::ParallelDistribution> parallel_distribution)
		: shape_(shape), dtype_(dtype), rpc_token_(rpc_token),
			parallel_desc_(parallel_desc), parallel_distribution_(parallel_distribution) {}

	~SendConsistencyAsyncRpcCtx() override {}

	Maybe<void> MakeDataBufferAndCallback(
			int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) override {
		const auto& flat_consistent_meta = std::make_shared<FlatConsistentTensorMeta>();	
		JUST(flat_consistent_meta->Init(shape_, dtype_, rpc_token_, parallel_distribution_));
		*buffer = flat_consistent_meta.get();
		*size = sizeof(FlatConsistentTensorMeta);
		*Callback = [flat_consistent_meta]() {};
		return Maybe<void>::Ok();
	}

 private:
	std::shared_ptr<const Shape> shape_;
	DataType dtype_;
	RpcToken rpc_token_;
	Symbol<ParallelDesc> parallel_desc_;
	Symbol<cfg::ParallelDistribution> parallel_distribution_;
};

Maybe<SendConsistencyAsyncRpcCtx> SendTensorMetaToNextRankInRing(const one::Tensor& tensor) {
	const auto& parallel_desc = JUST(tensor.parallel_desc());
	const auto& rank_ranges =
			JUST(SortedRankRanges::New4SoleDevicePerRankParallelDesc(parallel_desc));
  const auto& rpc_token = JUST(GetAutoIncrementalRpcToken(parallel_desc));
	std::shared_ptr<SendConsistencyAsyncRpcCtx> ctx;
	{
		const auto& shape = tensor.shape();
		DataType dtype = tensor.dtype();
		const RpcToken& rpc_token = JUST(tensor.rpc_token());
		const auto& parallel_distribution = JUST(tensor.parallel_distribution());
		ctx.reset(new SendConsistencyAsyncRpcCtx(
				shape, dtype, rpc_token, parallel_desc, parallel_distribution));
	}
	JUST(RpcUtil::SendToNextRankInRing(rank_ranges, rpc_token, ctx.get()));
	return ctx;
}

}

Maybe<CheckConsistencyAsyncRpcCtx> ReceiveTensorMetaFromPrevRankInRing(const one::Tensor& tensor) {
	const auto& parallel_desc = JUST(tensor.parallel_desc());
	const auto& rank_ranges =
			JUST(SortedRankRanges::New4SoleDevicePerRankParallelDesc(parallel_desc));
  const auto& rpc_token = JUST(GetAutoIncrementalRpcToken(parallel_desc));
	std::shared_ptr<CheckConsistencyAsyncRpcCtx> ctx;
	{
		const auto& shape = tensor.shape();
		DataType dtype = tensor.dtype();
		const RpcToken& rpc_token = JUST(tensor.rpc_token());
		const auto& parallel_distribution = JUST(tensor.parallel_distribution());
		ctx.reset(new CheckConsistencyAsyncRpcCtx(
				shape, dtype, rpc_token, parallel_desc, parallel_distribution));
	}
	JUST(RpcUtil::ReceiveFromPrevRankInRing(rank_ranges, rpc_token, ctx.get()));
	return ctx;
}

Maybe<CheckConsistencyAsyncRpcCtx> LaunchTensorMetaConsistencyCheck(const one::Tensor& tensor) {
	// no to wait send callback done. 
	JUST(SendTensorMetaToNextRankInRing(tensor));
	return ReceiveTensorMetaFromPrevRankInRing(tensor);
}

}
