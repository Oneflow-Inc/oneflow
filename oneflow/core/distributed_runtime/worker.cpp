#include "oneflow/core/distributed_runtime/worker.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common_runtime/runtime.h"
#include "oneflow/core/distributed_runtime/worker.pb.h"
#include "oneflow/core/job/runtime_context.h"

#include "tensorflow/core/lib/core/blocking_counter.h"

namespace oneflow {

Worker::Worker(
    int64_t this_machine_id, const std::string& this_machine_name,
    Network* data_net,
    const std::unordered_map<int64_t, std::shared_ptr<GrpcRemoteWorker>>&
        id2worker)
    : this_machine_id_(this_machine_id),
      this_machine_name_(this_machine_name),
      data_net_(data_net),
      id2worker_(id2worker) {}

Worker::~Worker() {}

::tensorflow::Status Worker::SendPlan(SendPlanRequest* request,
                                      SendPlanResponse* response,
                                      MyClosure done) {
  std::string str_plan;
  PrintProtoToString(request->plan(), &str_plan);
  LOG(INFO) << str_plan;

  ::oneflow::runtime::Runtime::Singleton()->SetPlan(request->plan());
  ::oneflow::runtime::Runtime::Singleton()->SetThisMachineName(
      this_machine_name_);

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerConnectDataPlane(
    WorkerConnectDataPlaneRequest* request,
    WorkerConnectDataPlaneResponse* response, MyClosure done) {
  data_net_->ConnectTopology();

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerInitRuntime(
    WorkerInitRuntimeRequest* request, WorkerInitRuntimeResponse* response,
    MyClosure done) {
  LOG(INFO) << "WorkerInitRuntime";

  ::oneflow::runtime::Runtime::Singleton()->InitRuntime();

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerInitModel(WorkerInitModelRequest* request,
                                             WorkerInitModelResponse* response,
                                             MyClosure done) {
  LOG(INFO) << "WorkerInitModel";
  ::oneflow::runtime::Runtime::Singleton()->InitModel();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerActivateActor(
    WorkerActivateActorRequest* request, WorkerActivateActorResponse* response,
    MyClosure done) {
  LOG(INFO) << "WorkerActivateActor";
  ::oneflow::runtime::Runtime::Singleton()->ActivateActor();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerSendRemoteRegst(
    WorkerSendRemoteRegstRequest* request,
    WorkerSendRemoteRegstResponse* response, MyClosure done) {
  if (request->ascending_order()) {
    LOG(INFO) << "WorkerSendRemoteRegst in ascending order";
  } else {
    LOG(INFO) << "WorkerSendRemoteRegst in descending order";
  }
  std::unordered_map<int64_t, std::vector<RemoteRegstDesc>>
      consumer2regst_descs;

  const std::vector<NetMemoryDescriptor>& local_net_memory_descs =
      RuntimeCtx::Singleton()->local_net_memory_descs();
  for (auto& net_memory_desc : local_net_memory_descs) {
    int index = 0;
    for (auto consumer_machine_id : net_memory_desc.consumer_machine_ids) {
      bool is_ascending = net_memory_desc.this_machine_id < consumer_machine_id;
      if (request->ascending_order() == is_ascending) {
        NetworkMemory* net_memory =
            static_cast<NetworkMemory*>(net_memory_desc.network_ptr);
        CHECK((uint64_t)net_memory_desc.local_ptr
              == net_memory->memory_discriptor().address);
        RemoteRegstDesc remote_regst_desc;
        remote_regst_desc.set_data_address(
            net_memory->memory_discriptor().address);
        remote_regst_desc.set_regst_address(
            (uint64_t)net_memory_desc.regst_ptr);
        remote_regst_desc.set_remote_token(
            net_memory->memory_discriptor().remote_token);
        remote_regst_desc.set_consumer_task_id(
            net_memory_desc.consumer_task_ids[index]);
        auto& regst_desc_it = consumer2regst_descs.find(consumer_machine_id);
        if (regst_desc_it == consumer2regst_descs.end()) {
          std::vector<RemoteRegstDesc> remote_regst_descs;
          remote_regst_descs.push_back(remote_regst_desc);
          consumer2regst_descs.insert(
              {consumer_machine_id, remote_regst_descs});
        } else {
          regst_desc_it->second.push_back(remote_regst_desc);
        }
      }
      ++index;
    }
  }
  int32_t rpc_count = consumer2regst_descs.size();
  if (rpc_count) {
    ::tensorflow::BlockingCounter blocking_counter(rpc_count);
    for (auto& pair : consumer2regst_descs) {
      struct Call {
        WorkerSendRemoteRegstToConsumerRequest req;
        WorkerSendRemoteRegstToConsumerResponse resp;
      };
      Call* call = new Call;
      call->req.set_producer_machine_id(this_machine_id_);
      call->req.set_consumer_machine_id(pair.first);

      for (auto& regst_desc : pair.second) {
        auto remote_regst_desc = call->req.add_remote_regst_descs();
        remote_regst_desc->set_data_address(regst_desc.data_address());
        remote_regst_desc->set_regst_address(
            regst_desc.regst_address());
        remote_regst_desc->set_remote_token(regst_desc.remote_token());
        remote_regst_desc->set_consumer_task_id(regst_desc.consumer_task_id());
      }

      auto cb = [call, pair, &blocking_counter](const ::tensorflow::Status& s) {
        if (s.ok()) {
          LOG(INFO) << "WorkerSendRemoteRegst OK: " << pair.first;
          blocking_counter.DecrementCount();
        } else {
          LOG(FATAL) << "WorkerSendRemoteRegst Fails to: " << pair.first;
        }
        delete call;
      };
      auto worker_it = id2worker_.find(pair.first);
      CHECK(worker_it != id2worker_.end());
      worker_it->second->WorkerSendRemoteRegstToConsumerAsync(&call->req,
                                                              &call->resp, cb);
    }
    blocking_counter.Wait();
  }

  LOG(INFO) << "Totally send out the number of RPCs: " << rpc_count;
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerSendRemoteRegstToConsumer(
    WorkerSendRemoteRegstToConsumerRequest* request,
    WorkerSendRemoteRegstToConsumerResponse* response, MyClosure done) {
  LOG(INFO) << "WorkerSendRemoteRegstToConsumer Recv remote regst descs";
  // ::oneflow::runtime::Runtime::Singleton()->SendRemoteRegstToDec();
  LOG(INFO) << "Worker machine id: " << this_machine_id_;
  LOG(INFO) << "Consumer machine id: " << request->consumer_machine_id();
  LOG(INFO) << "Producer machine id: " << request->producer_machine_id();
  int32_t regst_desc_num = request->remote_regst_descs_size();
  LOG(INFO) << "Remote regst desc num: " << regst_desc_num;
  for (int32_t i = 0; i < regst_desc_num; ++i) {
    LOG(INFO) << "Regst address: "
              << request->remote_regst_descs(i).regst_address();
    LOG(INFO) << "Data address:  "
              << request->remote_regst_descs(i).data_address();
    LOG(INFO) << "Remote token:  "
              << request->remote_regst_descs(i).remote_token();
    LOG(INFO) << "Consumer task_id: "
              << request->remote_regst_descs(i).consumer_task_id();
  }

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerStartActor(
    WorkerStartActorRequest* request, WorkerStartActorResponse* response,
    MyClosure done) {
  LOG(INFO) << "WorkerStartActor";
  ::oneflow::runtime::Runtime::Singleton()->StartActor();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerInitDataPlane(
    WorkerInitDataPlaneRequest* request, WorkerInitDataPlaneResponse* response,
    MyClosure done) {
  // data_net_->ConnectTopology();
  LOG(INFO) << "Worker Init DataPlane done";

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

}  // namespace oneflow
