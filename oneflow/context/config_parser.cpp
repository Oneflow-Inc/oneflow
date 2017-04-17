#include "context/config_parser.h"
#include <unordered_map>
#include <glog/logging.h>
#include "proto/oneflow.pb.h"
#include "proto/proto_io.h"
#include "context/solver_descriptor.h"
#include "context/machine_descriptor.h"
#include "context/net_descriptor.h"
#include "context/resource_descriptor.h"
#include "context/strategy_descriptor.h"

namespace oneflow {
ConfigParser::ConfigParser(const std::string& solver_name)
  : machine_descriptor_(nullptr),
    net_descriptor_(nullptr),
    resource_descriptor_(nullptr),
    strategy_descriptor_(nullptr) {

  oneflow::SolverProto solver;
  oneflow::ReadProtoFromTextFileOrDie(solver_name, &solver);

  CHECK(solver.has_machine_id());
  machine_descriptor_.reset(new MachineDescriptor(solver));

  solver_descriptor_.reset(new SolverDescriptor(solver));

  CHECK(solver.has_train_net());
  std::string train_net_name = solver.train_net();
  oneflow::NetParameter net_param;
  oneflow::ReadProtoFromTextFileOrDie(train_net_name, &net_param);
  net_descriptor_.reset(new NetDescriptor(net_param));

  CHECK(solver.has_resource());
  std::string resource_name = solver.resource();
  resource_descriptor_.reset(new ResourceDescriptor(resource_name));

  CHECK(solver.has_strategy());
  std::string strategy_name = solver.strategy();
  oneflow::Strategy strategy;
  oneflow::ReadProtoFromTextFileOrDie(strategy_name, &strategy);
  strategy_descriptor_.reset(
    new StrategyDescriptor(strategy, resource_descriptor_));
}
ConfigParser::~ConfigParser() {
}

std::shared_ptr<SolverDescriptor>
ConfigParser::solver_descriptor() const {
  return solver_descriptor_;
}

std::shared_ptr<MachineDescriptor>
  ConfigParser::machine_descriptor() const {
  return machine_descriptor_;
}
std::shared_ptr<NetDescriptor> ConfigParser::net_descriptor() const {
  return net_descriptor_;
}
std::shared_ptr<ResourceDescriptor> ConfigParser::resource_descriptor()
  const {
  return resource_descriptor_;
}
std::shared_ptr<StrategyDescriptor> ConfigParser::strategy_descriptor()
  const {
  return strategy_descriptor_;
}
void ConfigParser::set_strategy_descriptor(
  std::shared_ptr<StrategyDescriptor> strategy_descriptor) {
  strategy_descriptor_ = strategy_descriptor;
}
void ConfigParser::Validate() {
  CheckDeviceNumInGroup();
  CheckLayerOrderInGroup();
  // CheckDeviceSetInGroup();
  CheckGroupOrderInStrategy();
  CheckBalancedWorkLoadInStrategy();
}
void ConfigParser::CheckDeviceNumInGroup() {
  int32_t total_device_num = resource_descriptor_->total_device_num();
  int32_t group_num = strategy_descriptor_->group_num();
  for (int32_t id = 0; id < group_num; ++id) {
    auto device_set = strategy_descriptor_->device_set(id);
    for (int32_t did = 0; did < device_set.size(); ++did) {
      CHECK_GE(device_set[did], 0) << "Invalid device id in Resource file";
      CHECK_LT(device_set[did], total_device_num) << "Invalid device id in "
        << "Resource file";
    }
  }
}
void ConfigParser::CheckLayerOrderInGroup() {
  std::unordered_map<std::string, int32_t> layer_name2id;
  int32_t layer_num = net_descriptor_->layer_num();
  for (int32_t id = 0; id < layer_num; ++id) {
    auto layer_name = net_descriptor_->layer_name(id);
    layer_name2id.insert({layer_name, id});
  }

  int32_t group_num = strategy_descriptor_->group_num();
  for (int32_t sid = 0; sid < group_num; ++sid) {
    auto layer_set = strategy_descriptor_->layer_set(sid);
    auto layer_num = strategy_descriptor_->layer_num(sid);
    CHECK(layer_num != 0) << "Zero layer group";
    auto it = layer_name2id.find(layer_set[0]);
    CHECK(it != layer_name2id.end()) << "Unknown layer name";
    auto begin_id = it->second;
    for (int32_t lid = 1; lid < layer_num; ++lid) {
      it = layer_name2id.find(layer_set[lid]);
      CHECK(it != layer_name2id.find(layer_set[0])) << "Unknown layer name";
      auto current_id = it->second;
      ++begin_id;
      CHECK_EQ(current_id, begin_id) << "Non-consecutive layers in group";
    }
  }
}
/*
void ConfigParser::CheckDeviceSetInGroup() {
  // FIXME(jiyuan): Not support the relay from one device to another device
  int32_t device_num_per_machine = resource_descriptor_->device_num_per_machine();
  int32_t group_num = strategy_descriptor_->group_num();
  for (int32_t sid = 0; sid < group_num; ++sid) {
    auto device_set = strategy_descriptor_->device_set(sid);
    int32_t begin_id = device_set.front();
    int32_t end_id = device_set.back();
    CHECK_EQ(begin_id % device_num_per_machine, 0) << "A group not uses all the"
      << " provided devices of a machine";
    CHECK_EQ((end_id + 1) % device_num_per_machine, 0) << "A group not uses all"
      << " the provided devices of a machine";
  }
}
*/
void ConfigParser::CheckGroupOrderInStrategy() {
  std::unordered_map<std::string, int32_t> layer_name2id;
  int32_t layer_num = net_descriptor_->layer_num();
  for (int32_t id = 0; id < layer_num; ++id) {
    auto layer_name = net_descriptor_->layer_name(id);
    layer_name2id.insert({layer_name, id});
  }

  int32_t last_layer_id = 0;
  int32_t group_num = strategy_descriptor_->group_num();
  CHECK_GT(group_num, 0) << "Zero group number";
  for (int32_t sid = 1; sid < group_num; ++sid) {
    auto layer_set = strategy_descriptor_->layer_set(sid);
    auto layer_num = strategy_descriptor_->layer_num(sid);
    CHECK(layer_num != 0) << "Zero layer group";
    auto first_layer_name = layer_set.front();
    auto it = layer_name2id.find(first_layer_name);
    CHECK(it != layer_name2id.end()) << "Unknown layer name";
    int32_t first_layer_id = it->second;
    CHECK_EQ(last_layer_id + 1, first_layer_id) << "Non-consecutive groups";
    auto last_layer_name = layer_set.back();
    it = layer_name2id.find(last_layer_name);
    CHECK(it != layer_name2id.end()) << "Unknown layer name";
    last_layer_id = it->second;
  }
}
void ConfigParser::CheckBalancedWorkLoadInStrategy() {
  // TODO(jiyuan): 
}
}
