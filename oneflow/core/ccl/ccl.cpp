#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {
namespace ccl {

namespace {

Maybe<void> InitBroadcastRankHeap(std::vector<int64_t>* ranks, const ParallelDesc& parallel_desc, int64_t root) {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), parallel_desc.sorted_machine_ids().size());
  ranks->resize(parallel_desc.parallel_num());
  Optional<int64_t> root_index{};
  for (int64_t parallel_id = 0; parallel_id < parallel_desc.parallel_num(); ++parallel_id) {
    int64_t machine_id = JUST(parallel_desc.MachineId4ParallelId(parallel_id));
    if (machine_id == root) { root_index = parallel_id; }
    (*ranks)[parallel_id] = machine_id;
  }
  CHECK_OR_RETURN(root_index.has_value());
  {
    // swap 0 and root in ranks vector;
    int64_t root_idx = JUST(root_index.value());
    int64_t tmp = (*ranks)[0];
    (*ranks)[0] = (*ranks)[root_idx];
    (*ranks)[root_idx] = tmp;
  }
  return Maybe<void>::Ok();
}

}

template<>
Maybe<void> Broadcast<DeviceType::kCPU>(const char* in, char* out, size_t elem_cnt, DataType dtype, int64_t root, Symbol<ParallelDesc> parallel_desc, DeviceCtx* ctx) {
  CHECK_EQ_OR_RETURN(parallel_desc->device_type(), DeviceType::kCPU);
  static thread_local std::vector<int64_t> rank_heap{};
  InitBroadcastRankHeap(&rank_heap, *parallel_desc, root);
  RpcToken rpc_token = RpcToken::NewDataRpcToken();
  CHECK_OR_RETURN(IsPODDataType(dtype));
  size_t buffer_size = elem_cnt * GetSizeOfDataType(dtype);
  NaiveAsyncRpcCtx ctx(
    rpc_token,
    [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
      *buffer = (root == GlobalProcessCtx::Rank() ? const_cast<char*>(in): out);
      *size = buffer_size;
      *Cb = []{};
      return Maybe<void>::Ok();
    },
    [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
      *buffer = out;
      *size = buffer_size;
      *Cb = []{};
      return Maybe<void>::Ok();
    });
  const auto& rank_group = JUST(RankGroup::New(parallel_desc));
  JUST(RpcUtil::ReceiveDataFromParentInHeap(rank_heap, rpc_token, &ctx));
  JUST(RpcUtil::WaitUntilDoneOrTimeout(ctx, RpcUtil::TimeoutSeconds()));
  JUST(RpcUtil::SendDataToChildrenInHeap(rank_heap, rpc_token, &ctx));
  if (GlobalProcessCtx::Rank() == root) { std::memcpy(out, in, buffer_size); }
  JUST(RpcUtil::WaitUntilDoneOrTimeout(ctx, RpcUtil::TimeoutSeconds()));
  return Maybe<void>::Ok();
}

}
}
