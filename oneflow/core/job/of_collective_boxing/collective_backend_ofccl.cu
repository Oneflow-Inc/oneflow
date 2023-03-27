/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/job/of_collective_boxing/collective_backend_ofccl.h"
#include "oneflow/core/job/of_collective_boxing/of_request_store.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/graph/boxing/of_collective_boxing_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/shape.h"


#include <memory>
#include <utility>

namespace oneflow {

namespace boxing {

namespace of_collective {

namespace {

class CommRank final {
 public:
  OF_DISALLOW_COPY(CommRank);
  CommRank(int32_t device_id, int32_t global_rank, int32_t global_rank_count, int32_t local_rank,
           int32_t local_rank_count)
      : device_id_(device_id),
        global_rank_(global_rank),
        local_rank_(local_rank),
        nccl_comm_(nullptr) {}

  CommRank(CommRank&& rhs) noexcept {
    this->device_id_ = rhs.device_id_;
    this->global_rank_ = rhs.global_rank_;
    this->local_rank_ = rhs.local_rank_;
    this->nccl_comm_ = rhs.nccl_comm_;
    rhs.nccl_comm_ = nullptr;
  }

  ~CommRank() {
    if (nccl_comm_ != nullptr) {
      CudaCurrentDeviceGuard guard(device_id_);
      OF_NCCL_CHECK(ncclCommDestroy(nccl_comm_));
    }
  }

  int32_t device_id() const { return device_id_; }

  int32_t global_rank() const { return global_rank_; }

  ncclComm_t nccl_comm() const { return nccl_comm_; }

  void InitRank(ncclUniqueId unique_id, int32_t global_rank_count) {
    CudaCurrentDeviceGuard guard(device_id_);
    OF_NCCL_CHECK(ncclCommInitRank(&nccl_comm_, global_rank_count, unique_id, global_rank_));
  }

 private:
  int32_t device_id_;
  int32_t global_rank_;
  int32_t local_rank_;
  ncclComm_t nccl_comm_;
};

class CommGroup final {
 public:
  OF_DISALLOW_COPY(CommGroup);
  CommGroup() = default;
  ~CommGroup() = default;
  CommGroup(CommGroup&& rhs) noexcept {
    rank_vec_.swap(rhs.rank_vec_);
    global_rank_count_ = rhs.global_rank_count_;
  }

  void InitGroup(const DeviceSet& device_set, const std::string& unique_name) {
    CudaCurrentDeviceGuard guard;
    const int64_t this_machine_id = GlobalProcessCtx::Rank();
    global_rank_count_ = device_set.device_size();
    std::vector<int32_t> local_ranks;
    for (int32_t i = 0; i < global_rank_count_; ++i) {
      if (device_set.device(i).machine_id() == this_machine_id) { local_ranks.emplace_back(i); }
    }
    const int32_t local_rank_count = local_ranks.size();
    CHECK_GT(local_rank_count, 0);
    ncclUniqueId nccl_unique_id{};
    if (local_ranks.front() == 0) {
      OF_NCCL_CHECK(ncclGetUniqueId(&nccl_unique_id));
      if (local_rank_count != global_rank_count_) {
        Singleton<CtrlClient>::Get()->PushKV(unique_name, NcclUniqueIdToString(nccl_unique_id));
      }
    } else {
      Singleton<CtrlClient>::Get()->PullKV(unique_name, [&nccl_unique_id](const std::string& val) {
        NcclUniqueIdFromString(val, &nccl_unique_id);
      });
    }
    rank_vec_.reserve(local_rank_count);
    OF_NCCL_CHECK(ncclGroupStart());
    for (int32_t local_rank = 0; local_rank < local_ranks.size(); ++local_rank) {
      const int32_t global_rank = local_ranks.at(local_rank);
      const int32_t device_id = device_set.device(global_rank).device_id();
      OF_CUDA_CHECK(cudaSetDevice(device_id));
      rank_vec_.emplace_back(device_id, global_rank, global_rank_count_, local_rank,
                             local_rank_count);
      rank_vec_.at(local_rank).InitRank(nccl_unique_id, global_rank_count_);
    }
    OF_NCCL_CHECK(ncclGroupEnd());
  }

  int32_t global_rank_count() const { return global_rank_count_; }

  int32_t local_rank_count() const { return rank_vec_.size(); }

  const CommRank& GetCommRank(int32_t local_rank) const { return rank_vec_.at(local_rank); }

 private:
  std::vector<CommRank> rank_vec_;
  int32_t global_rank_count_ = 0;
};

struct DeviceSet7CommGroup {
  DeviceSet device_set;
  CommGroup comm_group;
};

std::string GetOfcclUniqueIdRpcKey(const std::string& name, int64_t coll_id) {
  return "CollectiveBoxingExecutorOfcclUniqueIdRpcKey-" + name + "-" + std::to_string(coll_id);
}

ncclRedOp_t OfcclGetReduceOp(ReduceMethod reduce_method) {
  if (reduce_method == kReduceMethodSum) {
    return ncclRedOp_t::ncclSum;
  } else {
    UNIMPLEMENTED();
    return ncclRedOp_t{};
  }
}

}  // namespace

struct CollectiveBackendOfccl::Impl {
  Impl(const CollectiveBoxingConf& conf, std::shared_ptr<OfRequestStore> request_store)
      : conf(conf), request_store(request_store) {
  }
  ~Impl() {
    coll_id2device_set7CommGroup.clear();
    device_id2ofccl_rank_ctx.clear();
  }

  void InitCommGroup(int64_t job_id) {
    request_store->ForEachMutOfRequestEntryInJob(
      job_id, [&](OfRequestEntry* request_entry, int32_t i, const OfRequestId& request_id) {
        const auto& request = request_entry->desc();
        if (request.op_desc().backend() != Backend::kBackendOFCCL) { return; }
        if (!request_entry->HasRankOnThisNode()) { return; }
        int coll_id = request_entry->coll_id();
        const DeviceSet& device_set = request.device_set();
        // 这里原来的结构是HashMap<DeviceSet, std::vector<CommGroup>> device_set2stream_id2comm_group;
        // 我们应该考虑coll_id了。
        // 给集合通信排coll_id应该发生在编译阶段。这时候multiclient的模式下，一个进程管一个rank，没法以全局视角分配coll_id
        // plan_util里，给集合通信指定的order应该可以充当coll_id。dependency_depth一次++，可以对应order的好几次++。
        // 创建了comm之后，要顺势完成prepare以及FinalizeRankCtx7StartHostThrds。
        // prepare需要comm参数。

        // 在每个rank上，每个集合通信都应该只被处理一次
        CHECK(coll_id2device_set7CommGroup.find(coll_id) == coll_id2device_set7CommGroup.end());
        coll_id2device_set7CommGroup[coll_id].device_set = device_set;
        // 我们需要为每一个集合通信都创建一个comm。
        coll_id2device_set7CommGroup[coll_id].comm_group.InitGroup(
          device_set,
          GetOfcclUniqueIdRpcKey(request.op_desc().name(), coll_id) // 不同rank上，同一个coll_id应该对应同样的op，会得到同样的rpc_key
        ); // 准备好了comm，接下来还需要count和datatype、op、rankCtx才可以进行prepare
        // 创建rankCtx之前，要set好device

        // VLOG(2) << "coll_id = " << coll_id << " local_rank_count = " << coll_id2device_set7CommGroup[coll_id].comm_group.local_rank_count();
        for (int32_t j = 0; j < coll_id2device_set7CommGroup[coll_id].comm_group.local_rank_count(); ++j) { // 其实是预期这里只有一个
          // 准备rank_ctx
          
          int local_device_id = coll_id2device_set7CommGroup[coll_id].comm_group.GetCommRank(j).device_id();

          OF_CUDA_CHECK(cudaSetDevice(local_device_id));

          // bugfix: 为支持多机pp，使用GlobalProcessCtx::Rank()
          // int curr_device_id = coll_id2device_set7CommGroup[coll_id].comm_group.GetCommRank(j).global_rank();
          int curr_device_id = GlobalProcessCtx::Rank();

          if (device_id2ofccl_rank_ctx.find(curr_device_id) == device_id2ofccl_rank_ctx.end()) {
            device_id2ofccl_rank_ctx[curr_device_id] = nullptr;
            ofcclInitRankCtx(&device_id2ofccl_rank_ctx[curr_device_id], local_device_id);
            VLOG(1) << "coll_id = " << coll_id << " curr_device_id = " << curr_device_id << " rankctx @ " << device_id2ofccl_rank_ctx[curr_device_id];
          }

          // 获取count和datatype、op信息。

          // oneflow/oneflow/core/common/shape.proto message ShapeProto
          // 根据ofccl/src/enqueue_ofccl.cc里 size_t channelSize = elem->count*ncclTypeSize(proxyOp->dtype)/elem->nChannels;的用法，可以认为count代表了元素个数，而不是字节数。
          size_t count = 1;
          const Shape shape = Shape(request.op_desc().shape());
          FOR_RANGE(int, shape_ax, 0, shape.NumAxes()) { count *= shape.At(shape_ax); }
          CHECK_GT(count, 0);
          // oneflow/oneflow/core/common/data_type.proto enum DataType
          ncclDataType_t nccl_data_type = GetNcclDataType(request.op_desc().data_type());
          ncclComm_t comm = coll_id2device_set7CommGroup[coll_id].comm_group.GetCommRank(j).nccl_comm();
          
          if (request.op_desc().op_type() == kOpTypeAllReduce) {
            // oneflow/oneflow/core/graph/boxing/of_collective_boxing.proto enum ReduceMethod
            ncclRedOp_t nccl_reduce_op = OfcclGetReduceOp(request.op_desc().reduce_method());
            OF_NCCL_CHECK(ofcclPrepareAllReduce(count, nccl_data_type, nccl_reduce_op, comm, coll_id, device_id2ofccl_rank_ctx[curr_device_id]));
          } else if (request.op_desc().op_type() == kOpTypeAllGather) {
            OF_NCCL_CHECK(ofcclPrepareAllGather(count, nccl_data_type, comm, coll_id, device_id2ofccl_rank_ctx[curr_device_id]));
          } else if (request.op_desc().op_type() == kOpTypeReduceScatter) {
            // oneflow/oneflow/core/graph/boxing/of_collective_boxing.proto enum ReduceMethod
            ncclRedOp_t nccl_reduce_op = OfcclGetReduceOp(request.op_desc().reduce_method());
            OF_NCCL_CHECK(ofcclPrepareReduceScatter(count, nccl_data_type, nccl_reduce_op, comm, coll_id, device_id2ofccl_rank_ctx[curr_device_id]));
          } else {
            UNIMPLEMENTED() << " request.op_desc().op_type() = " << request.op_desc().op_type();
          }
          
          // 只要没有一个rank被多个线程管理的情况，就不会发生同一个rank的信息会分裂到不同的rank_ctx实例中。
        }
      });
    return;
  }

  void FinalizeRankCtx7StartHostThrds() {
    for (auto &device_id7ofcll_rank_ctx : device_id2ofccl_rank_ctx) {
      ofcclFinalizeRankCtx7StartHostThrds(device_id7ofcll_rank_ctx.second);
    }
  }

  void Destroy() {
    for (auto &device_id7ofcll_rank_ctx : device_id2ofccl_rank_ctx) {
      VLOG(2) << "before ofcclDestroy in rank " << device_id7ofcll_rank_ctx.first;
      ofcclDestroy(device_id7ofcll_rank_ctx.second);
    }
  }

  ofcclRankCtx_t RetrieveOfcclRankCtx(int rank) {
    CHECK(device_id2ofccl_rank_ctx.find(rank) != device_id2ofccl_rank_ctx.end()) << " rank = " << rank;
    return device_id2ofccl_rank_ctx[rank];
  }

  CollectiveBoxingConf conf;
  std::shared_ptr<OfRequestStore> request_store;

  HashMap<int, DeviceSet7CommGroup> coll_id2device_set7CommGroup;
  HashMap<int, ofcclRankCtx_t> device_id2ofccl_rank_ctx;
};

CollectiveBackendOfccl::CollectiveBackendOfccl() = default;

CollectiveBackendOfccl::~CollectiveBackendOfccl() = default;

void CollectiveBackendOfccl::Init(std::shared_ptr<OfRequestStore> request_store) {
  // 我们复用了原来oneflow里的collective_boxing_conf
  VLOG(2) << "CollectiveBackendOfccl Init";
  impl_ = std::make_unique<Impl>(Singleton<ResourceDesc, ForSession>::Get()->collective_boxing_conf(),
                                 request_store);
}

void CollectiveBackendOfccl::InitJob(int64_t job_id) {
  CudaCurrentDeviceGuard guard;
  impl_->InitCommGroup(job_id); // 针对每个local rank创建了rank_ctx，并且对所有相关的集合通信执行了prepareColl（包括创建comm）
  // 接下来对每个local rank执行ofcclFinalizeRankCtx7StartHostThrds
  impl_->FinalizeRankCtx7StartHostThrds();
}

void CollectiveBackendOfccl::DeinitJob(int64_t job_id) {
  // 这个应该是最后退出执行要跑的，进行内存回收等等操作。
  VLOG(2) << "before CollectiveBackendOfccl impl_->Destroy()";
  impl_->Destroy();
}

ofcclRankCtx_t CollectiveBackendOfccl::RetrieveOfcclRankCtx(int rank) {
  return impl_->RetrieveOfcclRankCtx(rank);
}

}  // namespace of_collective

}  // namespace boxing

}  // namespace oneflow
