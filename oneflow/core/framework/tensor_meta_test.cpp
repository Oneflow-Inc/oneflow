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
#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace test {

namespace {
struct GlobalProcessCtxScope final {
  GlobalProcessCtxScope() {
    Global<ProcessCtx>::New();
    auto* ctx = Global<ProcessCtx>::Get();
    ctx->mutable_ctrl_addr()->Add();
    ctx->set_node_size(1);
  }
  ~GlobalProcessCtxScope() { Global<ProcessCtx>::Delete(); }
};
}  // namespace

TEST(TensorMeta, mirror_tensor_meta) {
  GlobalProcessCtxScope scope;
  Symbol<Device> device = CHECK_JUST(Device::New("cpu"));
  std::shared_ptr<Shape> shape(new Shape({2, 3, 4}));
  std::shared_ptr<one::TensorMeta> tensor_meta(
      new one::MirroredTensorMeta(shape, DataType::kInt32, device));
  ASSERT_TRUE(dynamic_cast<one::MirroredTensorMeta*>(tensor_meta.get()));
  ASSERT_FALSE(dynamic_cast<one::ConsistentTensorMeta*>(tensor_meta.get()));
  ASSERT_STREQ(tensor_meta->DebugString().c_str(),
               "MirroredTensorMeta(shape=(2,3,4), dtype=oneflow.int32, device=\"cpu:0\")");
}

TEST(TensorMeta, consistent_tensor_meta) {
  GlobalProcessCtxScope scope;
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0");
  parallel_conf.mutable_hierarchy()->add_dim(1);
  const auto& parallel_desc = ParallelDesc(parallel_conf);
  cfg::NdSbp nd_sbp;
  nd_sbp.add_sbp_parallel()->mutable_broadcast_parallel();
  std::shared_ptr<Shape> shape(new Shape({2, 3, 4}));
  std::shared_ptr<one::TensorMeta> tensor_meta(new one::ConsistentTensorMeta(
      shape, DataType::kInt32, SymbolOf(nd_sbp), SymbolOf(parallel_desc)));
  ASSERT_FALSE(dynamic_cast<one::MirroredTensorMeta*>(tensor_meta.get()));
  ASSERT_TRUE(dynamic_cast<one::ConsistentTensorMeta*>(tensor_meta.get()));
  ASSERT_STREQ(tensor_meta->DebugString().c_str(),
               "ConsistentTensorMeta(shape=(2,3,4), dtype=oneflow.int32, "
               "placement=oneflow.placement(device_type=\"cpu\", machine_device_ids={0 : [0]}, "
               "hierarchy=(1,)), Sbp=(oneflow.sbp.broadcast))");
}

}  // namespace test

}  // namespace oneflow