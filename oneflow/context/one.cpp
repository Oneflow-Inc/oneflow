#include "context/one.h"
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include "common/common.h"
#include "dag/node_meta.h"
#include "dag/logical_dag.h"
#include "dag/placement_group_dag.h"
#include "dag/segment_dag.h"
#include "dag/stage_dag.h"
#include "dag/pipe_dag.h"
#include "dag/actor_dag.h"
#include "dag/dag_iterator.h"
#include "layers/base_layer.h"
#include "context/id_map.h"
#include "context/machine_descriptor.h"
#include "context/strategy_descriptor.h"
#include "context/config_parser.h"
#include "context/resource_descriptor.h"
#include "context/net_descriptor.h"
//#include "thread/comm_bus.h"
//#include "thread/base_thread.h"
#include "task/job_manager.h"
#include "task/node_manager.h"
//#include "path/path_share_policy.h"
#include "path/path_manager.h"

namespace oneflow {
template <typename Dtype>
std::unique_ptr<TheOne<Dtype>> TheOne<Dtype>::singleton_;

template <typename Dtype>
std::shared_ptr<ConfigParser> TheOne<Dtype>::config_parser_;

template <typename Dtype>
std::shared_ptr<ConfigParser> TheOne<Dtype>::ps_config_parser_;

template <typename Dtype>
std::shared_ptr<IDMap> TheOne<Dtype>::id_map_;


template <typename Dtype>
std::shared_ptr<CommBus<Dtype>> TheOne<Dtype>::comm_bus_;

template <typename Dtype>
std::shared_ptr<PathManager<Dtype>> TheOne<Dtype>::path_manager_;

template <typename Dtype>
std::shared_ptr<JobManager<Dtype>> TheOne<Dtype>::job_manager_;

template <typename Dtype>
std::shared_ptr<NodeManager<Dtype>> TheOne<Dtype>::node_manager_;

//template <typename Dtype>
//std::shared_ptr<PathSharePolicy> TheOne<Dtype>::path_share_policy_;

template <typename Dtype>
TheOne<Dtype>::TheOne() {}

template <typename Dtype>
TheOne<Dtype>::~TheOne() {}

template <typename Dtype>
void TheOne<Dtype>::InitResource(const std::string& solver_name) {
  config_parser_.reset(new ConfigParser(solver_name));
  ps_config_parser_.reset(new ConfigParser(solver_name));
  id_map_.reset(new IDMap(config_parser_));
}

template <typename Dtype>
void TheOne<Dtype>::InitJob(const SolverProto& param) {
  CHECK(param.has_train_net());
  oneflow::NetParameter net_param;
  oneflow::ReadProtoFromTextFileOrDie(param.train_net(), &net_param);

  //logic_dag_.reset(new LogicalDag<Dtype>(config_parser_->net_descriptor(),
  //  PathType::kDataPath));
  //placement_group_dag_.reset(new PlacementGroupDag<Dtype>(
  //  logic_dag_, config_parser_->strategy_descriptor(), PathType::kDataPath));
  //segment_dag_.reset(new SegmentDag<Dtype>(
  //  logic_dag_, PathType::kDataPath));
  //segment_dag_->Build();
  //stage_dag_.reset(new StageDag<Dtype>(
  //  logic_dag_,
  //  segment_dag_, PathType::kDataPath));
  //pipe_dag_.reset(new PipeDag<Dtype>(
  //  segment_dag_,
  //  stage_dag_, PathType::kDataPath));
  //actor_dag_.reset(new ActorDag<Dtype>(
  //  logic_dag_,
  //  segment_dag_,
  //  stage_dag_,
  //  pipe_dag_, PathType::kDataPath));

  // NOTE(jiyuan): temporally, don't build PS DAG for single device setting
  //if (config_parser_->resource_descriptor()->total_device_num() > 1) {
  //  BuildPSDag();
  //}

  //job_manager_.reset(new JobManager<Dtype>(actor_dag_, ps_actor_dag_));
  //job_manager_->Init();

  //node_manager_.reset(new NodeManager<Dtype>(job_manager_));
  //node_manager_->Setup();
  //// Make sure to release all memory allocated by MemoryManager after
  //// computation
  //node_manager_->Release();
}

template <typename Dtype>
void TheOne<Dtype>::InitJob2(const SolverProto& param) {
  CHECK(param.has_train_net());
  path_manager_.reset(new PathManager<Dtype>());
  path_manager_->Initialize(param);
}

/*
template <typename Dtype>
void TheOne<Dtype>::InitThread() {
#if 0
  auto& machine_descriptor = config_parser_->machine_descriptor();
  comm_bus_.reset(new CommBus<Dtype>(machine_descriptor->total_thread_num()));
  comm_bus_->Init();   // Create message queues

  std::vector<BaseThread<Dtype>*> thread_vec_;
  // TODO(jiyuan):
  // 1, Create device threads, data thread, boxing thread, net thread
  // 2, Assign appropriate queue to the corresponding thread
  int32_t thread_local_id = 0;
  int32_t thread_id_size = machine_descriptor->device_thread_num();
  for (; thread_local_id < thread_id_size;
    ++thread_local_id) {
    BaseThread<Dtype>* device_thread
      = new BaseThread<Dtype>(comm_bus_->queue(thread_local_id));
    thread_vec_.push_back(device_thread);
  }
  thread_id_size += machine_descriptor->data_thread_num();
  for (; thread_local_id < thread_id_size; ++thread_local_id) {
    BaseThread<Dtype>* data_thread
      = new BaseThread<Dtype>(comm_bus_->queue(thread_local_id));
    thread_vec_.push_back(data_thread);
  }
  thread_id_size += machine_descriptor->boxing_thread_num();
  for (; thread_local_id < thread_id_size; ++thread_local_id) {
    BaseThread<Dtype>* boxing_thread
      = new BaseThread<Dtype>(comm_bus_->queue(thread_local_id));
    thread_vec_.push_back(boxing_thread);
  }
  thread_id_size += machine_descriptor->net_thread_num();
  for (; thread_local_id < thread_id_size; ++thread_local_id) {
    BaseThread<Dtype>* net_thread
      = new BaseThread<Dtype>(comm_bus_->queue(thread_local_id));
    thread_vec_.push_back(net_thread);
  }
#endif
}
template <typename Dtype>
void TheOne<Dtype>::InitNetwork() {
  Network* instance = GetNdspiRDMAInstance();
  int my_rank;       // TODO(feiga): set the value
  NetworkTopology topo;  // TODO(feiga): set the value
  instance->Init(my_rank, topo);
  instance->Barrier();

  // Exchange memory descriptor for every tasks' registers

  instance->Barrier();
}

*/
//INSTANTIATE_CLASS(TheOne);
}  // namespace oneflow
