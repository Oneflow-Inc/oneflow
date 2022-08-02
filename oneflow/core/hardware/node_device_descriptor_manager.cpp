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
#include "oneflow/core/hardware/node_device_descriptor_manager.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace hardware {

namespace {

std::string MakeNodeDeviceDescriptorRpcKey(const int64_t rank) {
  return "NodeDeviceDescriptorRpcKey/" + std::to_string(rank);
}

}  // namespace

struct NodeDeviceDescriptorManager::Impl {
  Impl(int64_t rank, int64_t num_ranks) : rank(rank) { nodes.resize(num_ranks); }
  std::vector<std::shared_ptr<const NodeDeviceDescriptor>> nodes;
  int64_t rank;
};

NodeDeviceDescriptorManager::NodeDeviceDescriptorManager() {
  impl_.reset(new Impl(GlobalProcessCtx::Rank(), GlobalProcessCtx::WorldSize()));
  std::shared_ptr<const NodeDeviceDescriptor> local = NodeDeviceDescriptor::Query();
  impl_->nodes.at(impl_->rank) = local;
  if (impl_->nodes.size() > 1) {
    std::string serialized_local_node;
    local->Serialize(&serialized_local_node);
    Singleton<CtrlClient>::Get()->PushKV(MakeNodeDeviceDescriptorRpcKey(impl_->rank),
                                         serialized_local_node);
    for (int64_t i = 0; i < impl_->nodes.size(); ++i) {
      if (i == impl_->rank) { continue; }
      Singleton<CtrlClient>::Get()->PullKV(
          MakeNodeDeviceDescriptorRpcKey(i), [&](const std::string& serialized) {
            impl_->nodes.at(i) = NodeDeviceDescriptor::Deserialize(serialized);
          });
    }
  }
}

NodeDeviceDescriptorManager::~NodeDeviceDescriptorManager() = default;

std::shared_ptr<const NodeDeviceDescriptor> NodeDeviceDescriptorManager::GetNodeDeviceDescriptor(
    int64_t rank) const {
  CHECK_LT(rank, impl_->nodes.size());
  return impl_->nodes.at(rank);
}

std::shared_ptr<const NodeDeviceDescriptor>
NodeDeviceDescriptorManager::GetLocalNodeDeviceDescriptor() const {
  return impl_->nodes.at(impl_->rank);
}

void NodeDeviceDescriptorManager::DumpSummary(const std::string& base) const {
  for (int64_t i = 0; i < impl_->nodes.size(); ++i) {
    impl_->nodes.at(i)->DumpSummary(JoinPath(base, "nodes", std::to_string(i)));
  }
}

}  // namespace hardware

}  // namespace oneflow
