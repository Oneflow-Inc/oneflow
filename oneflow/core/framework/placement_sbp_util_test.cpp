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
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace oneflow {
namespace test {

namespace {

struct GlobaProcessCtxScope final {
  GlobaProcessCtxScope(int64_t node_size, int64_t world_size) {
    Global<ProcessCtx>::New();
    auto* ctx = Global<ProcessCtx>::Get();
    for (int i = 0; i < world_size; ++i) { ctx->mutable_ctrl_addr()->Add(); }
    ctx->set_rank(0);
    ctx->set_node_size(node_size);
  }
  ~GlobaProcessCtxScope() { Global<ProcessCtx>::Delete(); }
};

}  // namespace

TEST(GetSelectedParallelIds, 1d_broadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size});
  std::vector<int> axis2is_selected{true};
  const auto& expected = std::vector<int64_t>{0, 1, 2, 3};
  for (int i = 0; i < parallel_size; ++i) {
    const auto& broadcast_parallel_ids =
        CHECK_JUST(private_details::GetSelectedParallelIds(hierarchy_shape, axis2is_selected, i));
    ASSERT_TRUE(*broadcast_parallel_ids == expected);
  }
}

TEST(GetSelectedParallelIds, 1d_nonbroadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size});
  std::vector<int> axis2is_selected{false};
  for (int i = 0; i < parallel_size; ++i) {
    const auto& broadcast_parallel_ids =
        CHECK_JUST(private_details::GetSelectedParallelIds(hierarchy_shape, axis2is_selected, i));
    const auto& expected = std::vector<int64_t>{i};
    ASSERT_TRUE(*broadcast_parallel_ids == expected);
  }
}

TEST(GetSelectedParallelIds, 2d_broadcast_broadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size, parallel_size});
  std::vector<int> axis2is_selected{true, true};
  std::vector<int64_t> expected{};
  for (int i = 0; i < parallel_size * parallel_size; ++i) { expected.push_back(i); }
  for (int i = 0; i < parallel_size * parallel_size; ++i) {
    const auto& broadcast_parallel_ids =
        CHECK_JUST(private_details::GetSelectedParallelIds(hierarchy_shape, axis2is_selected, i));
    ASSERT_TRUE(*broadcast_parallel_ids == expected);
  }
}

TEST(GetSelectedParallelIds, 2d_nonbroadcast_nonbroadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size, parallel_size});
  std::vector<int> axis2is_selected{false, false};
  for (int i = 0; i < parallel_size * parallel_size; ++i) {
    const auto& broadcast_parallel_ids =
        CHECK_JUST(private_details::GetSelectedParallelIds(hierarchy_shape, axis2is_selected, i));
    const auto& expected = std::vector<int64_t>{i};
    ASSERT_TRUE(*broadcast_parallel_ids == expected);
  }
}

TEST(GetSelectedParallelIds, 2d_broadcast_nonbroadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size, parallel_size});
  std::vector<int> axis2is_selected{true, false};
  for (int i = 0; i < parallel_size; ++i) {
    for (int j = 0; j < parallel_size; ++j) {
      std::vector<int64_t> expected{};
      for (int k = 0; k < parallel_size; ++k) { expected.push_back(k * parallel_size + j); }
      int64_t parallel_id = i * parallel_size + j;
      const auto& broadcast_parallel_ids = CHECK_JUST(
          private_details::GetSelectedParallelIds(hierarchy_shape, axis2is_selected, parallel_id));
      ASSERT_TRUE(*broadcast_parallel_ids == expected);
    }
  }
}

TEST(GetSelectedParallelIds, 2d_nonbroadcast_broadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size, parallel_size});
  std::vector<int> axis2is_selected{false, true};
  for (int i = 0; i < parallel_size; ++i) {
    std::vector<int64_t> expected{};
    for (int j = 0; j < parallel_size; ++j) { expected.push_back(i * parallel_size + j); }
    for (int j = 0; j < parallel_size; ++j) {
      int64_t parallel_id = i * parallel_size + j;
      const auto& broadcast_parallel_ids = CHECK_JUST(
          private_details::GetSelectedParallelIds(hierarchy_shape, axis2is_selected, parallel_id));
      ASSERT_TRUE(*broadcast_parallel_ids == expected);
    }
  }
}

namespace {

void InitSbpParallel(cfg::SbpParallel* sbp_parallel, const std::string& sbp_tag) {
  if (sbp_tag == "S0") {
    sbp_parallel->mutable_split_parallel()->set_axis(0);
  } else if (sbp_tag == "S1") {
    sbp_parallel->mutable_split_parallel()->set_axis(1);
  } else if (sbp_tag == "B") {
    sbp_parallel->mutable_broadcast_parallel();
  } else if (sbp_tag == "P") {
    sbp_parallel->mutable_partial_sum_parallel();
  } else {
    UNIMPLEMENTED();
  }
}

Symbol<cfg::ParallelDistribution> Get2dSbp(const std::string& sbp0, const std::string& sbp1) {
  cfg::ParallelDistribution nd_sbp;
  InitSbpParallel(nd_sbp.mutable_sbp_parallel()->Add(), sbp0);
  InitSbpParallel(nd_sbp.mutable_sbp_parallel()->Add(), sbp1);
  return SymbolOf(nd_sbp);
}

Symbol<one::ConsistentTensorMeta> MakeConsistentTensorMeta(
    Symbol<ParallelDesc> parallel_desc, Symbol<cfg::ParallelDistribution> nd_sbp) {
  const auto& shape = std::make_shared<const Shape>(DimVector{256, 256});
  one::ConsistentTensorMeta tensor_meta(shape, DataType::kInt32, nd_sbp, parallel_desc);
  return SymbolOf(tensor_meta);
}

}  // namespace

TEST(DecomposeByParallelId, decompose_axis0) {
  GlobaProcessCtxScope scope(2, 8);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  parallel_conf.add_device_name("1:0-3");
  parallel_conf.mutable_hierarchy()->add_dim(2);
  parallel_conf.mutable_hierarchy()->add_dim(4);
  const auto& parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
  const auto& src_nd_sbp = Get2dSbp("P", "B");
  const auto& dst_nd_sbp = Get2dSbp("S0", "B");
  for (int i = 0; i < 8; ++i) {
    const auto& tensor_meta = MakeConsistentTensorMeta(parallel_desc, src_nd_sbp);
    const auto& transformations =
        CHECK_JUST(private_details::DecomposeByParallelId(tensor_meta, dst_nd_sbp, i));
    ASSERT_EQ(transformations->size(), 1);
    ParallelConf expected_parallel_conf;
    expected_parallel_conf.set_device_tag("cpu");
    expected_parallel_conf.add_device_name(std::string("0:") + std::to_string(i % 4));
    expected_parallel_conf.add_device_name(std::string("1:") + std::to_string(i % 4));
    const auto& expected_parallel_desc = SymbolOf(ParallelDesc(expected_parallel_conf));
    ASSERT_TRUE(transformations->at(0).parallel_desc == expected_parallel_desc);
    ASSERT_EQ(transformations->at(0).src_nd_sbp->sbp_parallel_size(), 1);
    ASSERT_EQ(transformations->at(0).dst_nd_sbp->sbp_parallel_size(), 1);
    ASSERT_TRUE(transformations->at(0).src_nd_sbp->sbp_parallel(0).has_partial_sum_parallel());
    ASSERT_TRUE(transformations->at(0).dst_nd_sbp->sbp_parallel(0).has_split_parallel());
    ASSERT_EQ(transformations->at(0).dst_nd_sbp->sbp_parallel(0).split_parallel().axis(), 0);
  }
}

TEST(DecomposeByParallelId, decompose_axis1) {
  GlobaProcessCtxScope scope(2, 8);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  parallel_conf.add_device_name("1:0-3");
  parallel_conf.mutable_hierarchy()->add_dim(2);
  parallel_conf.mutable_hierarchy()->add_dim(4);
  const auto& parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
  const auto& src_nd_sbp = Get2dSbp("S0", "P");
  const auto& dst_nd_sbp = Get2dSbp("S0", "S1");
  for (int i = 0; i < 8; ++i) {
    const auto& tensor_meta = MakeConsistentTensorMeta(parallel_desc, src_nd_sbp);
    const auto& transformations =
        CHECK_JUST(private_details::DecomposeByParallelId(tensor_meta, dst_nd_sbp, i));
    ASSERT_EQ(transformations->size(), 1);
    ParallelConf expected_parallel_conf;
    expected_parallel_conf.set_device_tag("cpu");
    expected_parallel_conf.add_device_name(std::to_string(i / 4) + ":0-3");
    const auto& expected_parallel_desc = SymbolOf(ParallelDesc(expected_parallel_conf));
    ASSERT_TRUE(transformations->at(0).parallel_desc == expected_parallel_desc);
    ASSERT_EQ(transformations->at(0).src_nd_sbp->sbp_parallel_size(), 1);
    ASSERT_EQ(transformations->at(0).dst_nd_sbp->sbp_parallel_size(), 1);
    ASSERT_TRUE(transformations->at(0).src_nd_sbp->sbp_parallel(0).has_partial_sum_parallel());
    ASSERT_TRUE(transformations->at(0).dst_nd_sbp->sbp_parallel(0).has_split_parallel());
    ASSERT_EQ(transformations->at(0).dst_nd_sbp->sbp_parallel(0).split_parallel().axis(), 1);
  }
}

TEST(DecomposeByParallelId, decompose_two_axes) {
  GlobaProcessCtxScope scope(2, 8);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  parallel_conf.add_device_name("1:0-3");
  parallel_conf.mutable_hierarchy()->add_dim(2);
  parallel_conf.mutable_hierarchy()->add_dim(4);
  const auto& parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
  const auto& src_nd_sbp = Get2dSbp("S0", "P");
  const auto& dst_nd_sbp = Get2dSbp("B", "S0");
  for (int i = 0; i < 8; ++i) {
    const auto& tensor_meta = MakeConsistentTensorMeta(parallel_desc, src_nd_sbp);
    const auto& transformations =
        CHECK_JUST(private_details::DecomposeByParallelId(tensor_meta, dst_nd_sbp, i));
    ASSERT_EQ(transformations->size(), 2);
    {
      ParallelConf expected_parallel_conf;
      expected_parallel_conf.set_device_tag("cpu");
      expected_parallel_conf.add_device_name(std::string("0:") + std::to_string(i % 4));
      expected_parallel_conf.add_device_name(std::string("1:") + std::to_string(i % 4));
      const auto& expected_parallel_desc = SymbolOf(ParallelDesc(expected_parallel_conf));
      ASSERT_TRUE(transformations->at(0).parallel_desc == expected_parallel_desc);
      ASSERT_EQ(transformations->at(0).src_nd_sbp->sbp_parallel_size(), 1);
      ASSERT_EQ(transformations->at(0).dst_nd_sbp->sbp_parallel_size(), 1);
      ASSERT_TRUE(transformations->at(0).src_nd_sbp->sbp_parallel(0).has_split_parallel());
      ASSERT_TRUE(transformations->at(0).dst_nd_sbp->sbp_parallel(0).has_broadcast_parallel());
      ASSERT_EQ(transformations->at(0).src_nd_sbp->sbp_parallel(0).split_parallel().axis(), 0);
    }
    {
      ParallelConf expected_parallel_conf;
      expected_parallel_conf.set_device_tag("cpu");
      expected_parallel_conf.add_device_name(std::to_string(i / 4) + ":0-3");
      const auto& expected_parallel_desc = SymbolOf(ParallelDesc(expected_parallel_conf));
      ASSERT_TRUE(transformations->at(1).parallel_desc == expected_parallel_desc);
      ASSERT_EQ(transformations->at(1).src_nd_sbp->sbp_parallel_size(), 1);
      ASSERT_EQ(transformations->at(1).dst_nd_sbp->sbp_parallel_size(), 1);
      ASSERT_TRUE(transformations->at(1).src_nd_sbp->sbp_parallel(0).has_partial_sum_parallel());
      ASSERT_TRUE(transformations->at(1).dst_nd_sbp->sbp_parallel(0).has_split_parallel());
      ASSERT_EQ(transformations->at(1).dst_nd_sbp->sbp_parallel(0).split_parallel().axis(), 0);
    }
  }
}

}  // namespace test
}  // namespace oneflow
