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
#include "gtest/gtest.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/common/tensor_meta.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {
namespace test {

namespace {

struct GlobaProcessCtxScope final {
  GlobaProcessCtxScope(GlobaProcessCtxScope&) = default;
  GlobaProcessCtxScope(GlobaProcessCtxScope&&) = default;
  GlobaProcessCtxScope& operator=(GlobaProcessCtxScope&) = default;
  GlobaProcessCtxScope& operator=(GlobaProcessCtxScope&&) = default;
  GlobaProcessCtxScope(int64_t node_size, int64_t world_size) {
    Singleton<ProcessCtx>::New();
    auto* ctx = Singleton<ProcessCtx>::Get();
    for (int i = 0; i < world_size; ++i) { ctx->mutable_ctrl_addr()->Add(); }
    ctx->set_rank(0);
    ctx->set_node_size(node_size);
  }
  ~GlobaProcessCtxScope() { Singleton<ProcessCtx>::Delete(); }
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
  for (int i = 0; i < parallel_size * parallel_size; ++i) { expected.emplace_back(i); }
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
      for (int k = 0; k < parallel_size; ++k) { expected.emplace_back(k * parallel_size + j); }
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
    for (int j = 0; j < parallel_size; ++j) { expected.emplace_back(i * parallel_size + j); }
    for (int j = 0; j < parallel_size; ++j) {
      int64_t parallel_id = i * parallel_size + j;
      const auto& broadcast_parallel_ids = CHECK_JUST(
          private_details::GetSelectedParallelIds(hierarchy_shape, axis2is_selected, parallel_id));
      ASSERT_TRUE(*broadcast_parallel_ids == expected);
    }
  }
}

namespace {

void InitSbpParallel(SbpParallel* sbp_parallel, const std::string& sbp_tag) {
  CHECK(sbp_tag.size() == 1 || sbp_tag.size() == 2);
  if (sbp_tag[0] == 'S') {
    CHECK_EQ(sbp_tag.size(), 2);
    int64_t axis = sbp_tag[1] - '0';
    sbp_parallel->mutable_split_parallel()->set_axis(axis);
  } else if (sbp_tag == "B") {
    sbp_parallel->mutable_broadcast_parallel();
  } else if (sbp_tag == "P") {
    sbp_parallel->mutable_partial_sum_parallel();
  } else {
    UNIMPLEMENTED();
  }
}

template<typename... Args>
Symbol<NdSbp> GetNdSbp(Args... sbps) {
  NdSbp nd_sbp;
  for (const auto& sbp : std::vector<std::string>{sbps...}) {
    InitSbpParallel(nd_sbp.mutable_sbp_parallel()->Add(), sbp);
  }
  return SymbolOf(nd_sbp);
}

Symbol<one::GlobalTensorMeta> MakeGlobalTensorMeta(Symbol<ParallelDesc> parallel_desc,
                                                   Symbol<NdSbp> nd_sbp) {
  auto shape = Shape(DimVector{256, 256});
  one::GlobalTensorMeta tensor_meta(shape, DataType::kInt32, nd_sbp, parallel_desc);
  return SymbolOf(tensor_meta);
}

}  // namespace

TEST(DecomposeIntoNaiveTransformations, decompose_axis0) {
  GlobaProcessCtxScope scope(2, 8);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  parallel_conf.add_device_name("1:0-3");
  parallel_conf.mutable_hierarchy()->add_dim(2);
  parallel_conf.mutable_hierarchy()->add_dim(4);
  const auto& parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
  const auto& src_nd_sbp = GetNdSbp("P", "B");
  const auto& dst_nd_sbp = GetNdSbp("S0", "B");
  const auto& tensor_meta = MakeGlobalTensorMeta(parallel_desc, src_nd_sbp);
  const auto& transformations =
      CHECK_JUST(private_details::DecomposeIntoNaiveTransformations(tensor_meta, dst_nd_sbp));
  ASSERT_EQ(transformations->size(), 1);
  ParallelConf expected_parallel_conf;
  expected_parallel_conf.set_device_tag("cpu");
  expected_parallel_conf.add_device_name(std::string("0:0"));
  expected_parallel_conf.add_device_name(std::string("1:0"));
  const auto& expected_parallel_desc = SymbolOf(ParallelDesc(expected_parallel_conf));
  const auto& ctensor_meta = transformations->at(0).global_tensor_meta;
  ASSERT_TRUE(ctensor_meta->parallel_desc() == expected_parallel_desc);
  ASSERT_EQ(ctensor_meta->nd_sbp()->sbp_parallel_size(), 1);
  ASSERT_EQ(transformations->at(0).dst_nd_sbp->sbp_parallel_size(), 1);
  ASSERT_TRUE(ctensor_meta->nd_sbp()->sbp_parallel(0).has_partial_sum_parallel());
  ASSERT_TRUE(transformations->at(0).dst_nd_sbp->sbp_parallel(0).has_split_parallel());
  ASSERT_EQ(transformations->at(0).dst_nd_sbp->sbp_parallel(0).split_parallel().axis(), 0);
}

TEST(DecomposeIntoNaiveTransformations, decompose_axis1) {
  GlobaProcessCtxScope scope(2, 8);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-3");
  parallel_conf.add_device_name("1:0-3");
  parallel_conf.mutable_hierarchy()->add_dim(2);
  parallel_conf.mutable_hierarchy()->add_dim(4);
  const auto& parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
  const auto& src_nd_sbp = GetNdSbp("S0", "P");
  const auto& dst_nd_sbp = GetNdSbp("S0", "S1");
  const auto& tensor_meta = MakeGlobalTensorMeta(parallel_desc, src_nd_sbp);
  const auto& transformations =
      CHECK_JUST(private_details::DecomposeIntoNaiveTransformations(tensor_meta, dst_nd_sbp));
  ASSERT_EQ(transformations->size(), 1);
  ParallelConf expected_parallel_conf;
  expected_parallel_conf.set_device_tag("cpu");
  expected_parallel_conf.add_device_name("0:0-3");
  const auto& expected_parallel_desc = SymbolOf(ParallelDesc(expected_parallel_conf));
  const auto& ctensor_meta = transformations->at(0).global_tensor_meta;
  ASSERT_TRUE(ctensor_meta->parallel_desc() == expected_parallel_desc);
  ASSERT_EQ(ctensor_meta->nd_sbp()->sbp_parallel_size(), 1);
  ASSERT_EQ(transformations->at(0).dst_nd_sbp->sbp_parallel_size(), 1);
  ASSERT_TRUE(ctensor_meta->nd_sbp()->sbp_parallel(0).has_partial_sum_parallel());
  ASSERT_TRUE(transformations->at(0).dst_nd_sbp->sbp_parallel(0).has_split_parallel());
  ASSERT_EQ(transformations->at(0).dst_nd_sbp->sbp_parallel(0).split_parallel().axis(), 1);
}

TEST(DecomposeIntoNaiveTransformations, decompose_two_axes) {
  GlobaProcessCtxScope scope(2, 8);
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0-1");
  parallel_conf.add_device_name("1:0-1");
  parallel_conf.mutable_hierarchy()->add_dim(2);
  parallel_conf.mutable_hierarchy()->add_dim(2);
  const auto& parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
  const auto& src_nd_sbp = GetNdSbp("S0", "P");
  const auto& dst_nd_sbp = GetNdSbp("B", "S0");
  const auto& tensor_meta = MakeGlobalTensorMeta(parallel_desc, src_nd_sbp);
  const auto& transformations =
      CHECK_JUST(private_details::DecomposeIntoNaiveTransformations(tensor_meta, dst_nd_sbp));
  ASSERT_EQ(transformations->size(), 2);
  {
    ParallelConf expected_parallel_conf;
    expected_parallel_conf.set_device_tag("cpu");
    expected_parallel_conf.add_device_name(std::string("0:0"));
    expected_parallel_conf.add_device_name(std::string("1:0"));
    const auto& expected_parallel_desc = SymbolOf(ParallelDesc(expected_parallel_conf));
    const auto& ctensor_meta = transformations->at(0).global_tensor_meta;
    ASSERT_TRUE(ctensor_meta->parallel_desc() == expected_parallel_desc);
    ASSERT_EQ(ctensor_meta->nd_sbp()->sbp_parallel_size(), 1);
    ASSERT_EQ(transformations->at(0).dst_nd_sbp->sbp_parallel_size(), 1);
    ASSERT_TRUE(ctensor_meta->nd_sbp()->sbp_parallel(0).has_split_parallel());
    ASSERT_TRUE(transformations->at(0).dst_nd_sbp->sbp_parallel(0).has_broadcast_parallel());
    ASSERT_EQ(ctensor_meta->nd_sbp()->sbp_parallel(0).split_parallel().axis(), 0);
  }
  {
    ParallelConf expected_parallel_conf;
    expected_parallel_conf.set_device_tag("cpu");
    expected_parallel_conf.add_device_name("0:0-1");
    const auto& expected_parallel_desc = SymbolOf(ParallelDesc(expected_parallel_conf));
    const auto& ctensor_meta = transformations->at(1).global_tensor_meta;
    ASSERT_TRUE(ctensor_meta->parallel_desc() == expected_parallel_desc);
    ASSERT_EQ(ctensor_meta->nd_sbp()->sbp_parallel_size(), 1);
    ASSERT_EQ(transformations->at(1).dst_nd_sbp->sbp_parallel_size(), 1);
    ASSERT_TRUE(ctensor_meta->nd_sbp()->sbp_parallel(0).has_partial_sum_parallel());
    ASSERT_TRUE(transformations->at(1).dst_nd_sbp->sbp_parallel(0).has_split_parallel());
    ASSERT_EQ(transformations->at(1).dst_nd_sbp->sbp_parallel(0).split_parallel().axis(), 0);
  }
}

TEST(CalcDecomposableEquivalentShapeAndNdSbpPair, naive) {
  Shape shape(DimVector{4, 4});
  Shape hierarchy(DimVector{4, 4});
  const auto& src_nd_sbp = GetNdSbp("S0", "S1");
  const auto& dst_nd_sbp = GetNdSbp("B", "P");
  const auto& maybe_tuple = TRY(private_details::CalcDecomposableEquivalentShapeAndNdSbpPair(
      shape, hierarchy, src_nd_sbp, dst_nd_sbp));
  ASSERT_TRUE(maybe_tuple.IsOk());
  const auto& tuple = CHECK_JUST(maybe_tuple);
  ASSERT_TRUE(*std::get<0>(*tuple) == shape);
  ASSERT_TRUE(std::get<1>(*tuple) == src_nd_sbp);
  ASSERT_TRUE(std::get<2>(*tuple) == dst_nd_sbp);
}

TEST(CalcDecomposableEquivalentShapeAndNdSbpPair, expand_src) {
  Shape shape(DimVector{16, 4});
  Shape hierarchy(DimVector{4, 4});
  const auto& src_nd_sbp = GetNdSbp("S0", "S0");
  const auto& dst_nd_sbp = GetNdSbp("B", "P");
  const auto& maybe_tuple = TRY(private_details::CalcDecomposableEquivalentShapeAndNdSbpPair(
      shape, hierarchy, src_nd_sbp, dst_nd_sbp));
  ASSERT_TRUE(maybe_tuple.IsOk());
  const auto& tuple = CHECK_JUST(maybe_tuple);
  ASSERT_TRUE(*std::get<0>(*tuple) == Shape(DimVector{4, 4, 4}));
  ASSERT_TRUE(std::get<1>(*tuple) == GetNdSbp("S0", "S1"));
  ASSERT_TRUE(std::get<2>(*tuple) == dst_nd_sbp);
}

TEST(CalcDecomposableEquivalentShapeAndNdSbpPair, expand_failed) {
  Shape shape(DimVector{32, 4});
  Shape hierarchy(DimVector{4, 4, 4});
  const auto& src_nd_sbp = GetNdSbp("S0", "S0", "S0");
  const auto& dst_nd_sbp = GetNdSbp("P", "S0", "S1");
  const auto& maybe_tuple = TRY(private_details::CalcDecomposableEquivalentShapeAndNdSbpPair(
      shape, hierarchy, src_nd_sbp, dst_nd_sbp));
  ASSERT_FALSE(maybe_tuple.IsOk());
}

TEST(IsNdSbpBoxingAcyclic, yes) {
  const auto& src_nd_sbp = GetNdSbp("S0", "S1", "S2");
  const auto& dst_nd_sbp = GetNdSbp("S1", "S2", "S3");
  const auto& maybe_acyclic = TRY(private_details::IsNdSbpBoxingAcyclic(src_nd_sbp, dst_nd_sbp));
  ASSERT_TRUE(maybe_acyclic.IsOk());
  ASSERT_TRUE(CHECK_JUST(maybe_acyclic));
}

TEST(IsNdSbpBoxingAcyclic, ring) {
  const auto& src_nd_sbp = GetNdSbp("S0", "S1", "S2");
  const auto& dst_nd_sbp = GetNdSbp("S1", "S2", "S0");
  const auto& maybe_acyclic = TRY(private_details::IsNdSbpBoxingAcyclic(src_nd_sbp, dst_nd_sbp));
  ASSERT_TRUE(maybe_acyclic.IsOk());
  ASSERT_FALSE(CHECK_JUST(maybe_acyclic));
}

TEST(IsNdSbpBoxingAcyclic, partial_ring) {
  const auto& src_nd_sbp = GetNdSbp("B", "S0", "S1", "S2", "S5");
  const auto& dst_nd_sbp = GetNdSbp("P", "S1", "S2", "S0", "S4");
  const auto& maybe_acyclic = TRY(private_details::IsNdSbpBoxingAcyclic(src_nd_sbp, dst_nd_sbp));
  ASSERT_TRUE(maybe_acyclic.IsOk());
  ASSERT_FALSE(CHECK_JUST(maybe_acyclic));
}

TEST(IsNdSbpBoxingAcyclic, dag) {
  const auto& src_nd_sbp = GetNdSbp("S0", "S1", "S2");
  const auto& dst_nd_sbp = GetNdSbp("S1", "S2", "S3");
  const auto& maybe_acyclic = TRY(private_details::IsNdSbpBoxingAcyclic(src_nd_sbp, dst_nd_sbp));
  ASSERT_TRUE(maybe_acyclic.IsOk());
  ASSERT_TRUE(CHECK_JUST(maybe_acyclic));
}

TEST(GetNdSbpValidTransformationAxisSequence, naive) {
  const auto& src_nd_sbp = GetNdSbp("S0", "S1", "S2");
  const auto& dst_nd_sbp = GetNdSbp("S0", "B", "S2");
  const auto& maybe_axis_seq =
      TRY(private_details::GetNdSbpValidTransformationAxisSequence(src_nd_sbp, dst_nd_sbp));
  ASSERT_TRUE(maybe_axis_seq.IsOk());
  const auto& axis_seq = CHECK_JUST(maybe_axis_seq);
  ASSERT_TRUE(*axis_seq == std::vector<int64_t>{1});
}

TEST(GetNdSbpValidTransformationAxisSequence, 2d) {
  const auto& src_nd_sbp = GetNdSbp("B", "S0");
  const auto& dst_nd_sbp = GetNdSbp("S0", "S1");
  const auto& maybe_axis_seq =
      TRY(private_details::GetNdSbpValidTransformationAxisSequence(src_nd_sbp, dst_nd_sbp));
  ASSERT_TRUE(maybe_axis_seq.IsOk());
  const auto& axis_seq = CHECK_JUST(maybe_axis_seq);
  ASSERT_TRUE(*axis_seq == (std::vector<int64_t>{1, 0}));
}

TEST(GetNdSbpValidTransformationAxisSequence, 3d) {
  const auto& src_nd_sbp = GetNdSbp("S0", "S1", "S2");
  const auto& dst_nd_sbp = GetNdSbp("S1", "S2", "S3");
  const auto& maybe_axis_seq =
      TRY(private_details::GetNdSbpValidTransformationAxisSequence(src_nd_sbp, dst_nd_sbp));
  ASSERT_TRUE(maybe_axis_seq.IsOk());
  const auto& axis_seq = CHECK_JUST(maybe_axis_seq);
  ASSERT_TRUE(*axis_seq == (std::vector<int64_t>{2, 1, 0}));
}

}  // namespace test
}  // namespace oneflow
