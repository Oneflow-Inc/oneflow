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
#include <gtest/gtest.h>
#include "oneflow/core/ep/test/primitive/primitive_test.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/primitive/gather.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace oneflow {

namespace ep {
namespace primitive {

namespace test {

namespace {

struct TestGatherToolPack {
  using StreamGuard = ep::test::StreamGuard;
  std::unique_ptr<Memcpy> h2d;
  std::unique_ptr<Memcpy> d2h;
  StreamGuard* stream;

  TestGatherToolPack(DeviceType device_type, std::shared_ptr<Device>& device) {
    h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    assert(d2h.operator bool());
    assert(h2d.operator bool());
    stream = new StreamGuard(device.get());
  }

  ~TestGatherToolPack() { delete stream; }
};

template<typename ParamsType, typename IndicesType>
struct TestGatherTensorPack {
  using InOutTensor = Eigen::Tensor<ParamsType, 4, Eigen::RowMajor>;
  using IndicesTensor = Eigen::Tensor<IndicesType, 2, Eigen::RowMajor>;
  using PinnedMemoryGuard = ep::test::PinnedMemoryGuard;
  using DeviceMemoryGuard = ep::test::DeviceMemoryGuard;

  size_t batch_dim_size;
  size_t outer_dim_size;
  size_t gather_dim_size;
  size_t inner_dim_size;
  size_t offset;
  size_t per_batch_indices_size;
  size_t params_size;
  size_t indices_size;
  size_t out_size;

  InOutTensor* params;
  IndicesTensor* indices;
  InOutTensor* out;

  PinnedMemoryGuard* host_params;
  PinnedMemoryGuard* host_indices;
  PinnedMemoryGuard* host_out;
  DeviceMemoryGuard* device_params;
  DeviceMemoryGuard* device_indices;
  DeviceMemoryGuard* device_out;

  TestGatherTensorPack(size_t batch_dim_size, size_t outer_dim_size, size_t gather_dim_size,
                       size_t inner_dim_size, size_t per_batch_indices_size, size_t offset)
      : batch_dim_size(batch_dim_size),
        outer_dim_size(outer_dim_size),
        gather_dim_size(gather_dim_size),
        inner_dim_size(inner_dim_size),
        offset(offset),
        per_batch_indices_size(per_batch_indices_size) {
    params = new InOutTensor(batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size);
    indices = new IndicesTensor(batch_dim_size, per_batch_indices_size);
    out = new InOutTensor(batch_dim_size, outer_dim_size, per_batch_indices_size, inner_dim_size);

    params->setRandom();
    out->setZero();
    IndicesType* indices_data = indices->data();
    for (size_t i = 0; i < batch_dim_size * per_batch_indices_size; i++) {
      indices_data[i] = std::rand() % gather_dim_size;
    }
  }

  ~TestGatherTensorPack() {
    delete params;
    delete indices;
    delete out;
    if (host_params != nullptr) delete host_params;
    if (host_indices != nullptr) delete host_indices;
    if (host_out != nullptr) delete host_out;
    if (device_params != nullptr) delete device_params;
    if (device_indices != nullptr) delete device_indices;
    if (device_out != nullptr) delete device_out;
  }

  void AllocateMemory(std::shared_ptr<Device>& device) {
    params_size = batch_dim_size * outer_dim_size * gather_dim_size * inner_dim_size;
    indices_size = batch_dim_size * per_batch_indices_size;
    out_size = batch_dim_size * outer_dim_size * per_batch_indices_size * inner_dim_size;

    host_params = new PinnedMemoryGuard(device.get(), params_size * sizeof(ParamsType));
    host_indices = new PinnedMemoryGuard(device.get(), indices_size * sizeof(IndicesType));
    host_out = new PinnedMemoryGuard(device.get(), out_size * sizeof(ParamsType));
    device_params = new DeviceMemoryGuard(device.get(), params_size * sizeof(ParamsType));
    device_indices = new DeviceMemoryGuard(device.get(), indices_size * sizeof(IndicesType));
    device_out = new DeviceMemoryGuard(device.get(), out_size * sizeof(ParamsType));
  }

  void CopySrcDataToPinnedMemory(std::shared_ptr<Device>& device) {
    std::memcpy(host_params->ptr(), params->data(), params_size * sizeof(ParamsType));
    std::memcpy(host_indices->ptr(), indices->data(), indices_size * sizeof(IndicesType));
  }

  void CopySrcDataToDevice(TestGatherToolPack* tool) {
    tool->h2d->Launch(tool->stream->stream(), device_params->ptr<ParamsType>(),
                      host_params->ptr<ParamsType>(), params_size * sizeof(ParamsType));
    tool->h2d->Launch(tool->stream->stream(), device_indices->ptr<ParamsType>(),
                      host_indices->ptr<ParamsType>(), indices_size * sizeof(ParamsType));
  }

  void CopyDstDataToPinnedMemory(TestGatherToolPack* tool) {
    tool->d2h->Launch(tool->stream->stream(), host_out->ptr<ParamsType>(),
                      device_out->ptr<ParamsType>(), out_size * sizeof(ParamsType));
  }

  bool VerifyResult() {
    NdIndexOffsetHelper<size_t, 4> in_helper(batch_dim_size, outer_dim_size, gather_dim_size,
                                             inner_dim_size);
    NdIndexOffsetHelper<size_t, 4> out_helper(batch_dim_size, outer_dim_size,
                                              indices_size / batch_dim_size, inner_dim_size);
    NdIndexOffsetHelper<size_t, 2> indices_helper(batch_dim_size, indices_size / batch_dim_size);
    size_t index[4];
    size_t indices_index[2];
    for (size_t i = 0; i < out_size; i++) {
      out_helper.OffsetToNdIndex(i, index);
      indices_index[0] = index[0];  // batch_dim_index
      indices_index[1] = index[2];  // gather_dim_index
      index[2] =
          host_indices->ptr<IndicesType>()[indices_helper.NdIndexToOffset(indices_index)] - offset;
      if (index[2] >= 0 && index[2] <= gather_dim_size) {
        ParamsType correct_result =
            host_params->ptr<ParamsType>()[in_helper.NdIndexToOffset(index)];
        if (correct_result != host_out->ptr<ParamsType>()[i]) return false;
      } else {
        if (host_out->ptr<ParamsType>()[i] != 0) return false;
      }
    }

    return true;
  }
};

template<typename T>
std::vector<std::vector<T>> CartProduct(const std::vector<std::vector<T>>& v) {
  std::vector<std::vector<T>> s = {{}};
  for (const auto& u : v) {
    std::vector<std::vector<T>> r;
    for (const auto& x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = std::move(r);
  }
  return s;
}

template<DataType params_type, typename ParamsType, DataType indices_type, typename IndicesType>
void TestGather(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  std::vector<std::vector<int>> test_set = {
      {16},          // batch_dim_size
      {16},          // outer_dim_size
      {16},          // gather_dim_size
      {16, 32, 41, 53},  // inner_dim_size
      {32},          // per_batch_indices_size
      {0, 4},        // offset
  };

  std::vector<std::vector<int>> test_shapes = CartProduct<int>(test_set);

  for (const auto& device_type : device_types) {
    auto device = registry->GetDevice(device_type, 0);

    TestGatherToolPack* test_tool_pack = new TestGatherToolPack(device_type, device);

    std::unique_ptr<Gather> gather =
        NewPrimitive<GatherFactory>(device_type, params_type, indices_type);
    ASSERT_TRUE(gather.operator bool());

#define TEST_GATHER_LOG_OUTPUT true

    for (const auto& e : test_shapes) {
#if TEST_GATHER_LOG_OUTPUT
      std::cout << ">>>>>>>>>>>>>>> TEST START <<<<<<<<<<<<<<<" << std::endl;
      std::cout << "device_type:              " << (device_type == 1 ? "cpu" : "cuda") << std::endl;
      std::cout << "bacth_dim_size:           " << e[0] << std::endl;
      std::cout << "outer_dim_size:           " << e[1] << std::endl;
      std::cout << "gather_dim_size:          " << e[2] << std::endl;
      std::cout << "inner_dim_size:           " << e[3] << std::endl;
      std::cout << "per_batch_indices_size:   " << e[4] << std::endl;
      std::cout << "offset:                   " << e[5] << std::endl;
#endif
      TestGatherTensorPack<ParamsType, IndicesType>* test_tensor_pack =
          new TestGatherTensorPack<ParamsType, IndicesType>(e[0], e[1], e[2], e[3], e[4], e[5]);

      test_tensor_pack->AllocateMemory(device);
      test_tensor_pack->CopySrcDataToPinnedMemory(device);
      test_tensor_pack->CopySrcDataToDevice(test_tool_pack);

      gather->Launch(test_tool_pack->stream->stream(), test_tensor_pack->batch_dim_size,
                     test_tensor_pack->outer_dim_size, test_tensor_pack->gather_dim_size,
                     test_tensor_pack->inner_dim_size, test_tensor_pack->offset,
                     test_tensor_pack->device_params->template ptr<ParamsType>(),
                     test_tensor_pack->indices_size,
                     test_tensor_pack->device_indices->template ptr<IndicesType>(),
                     test_tensor_pack->device_out->template ptr<ParamsType>());
      test_tensor_pack->CopyDstDataToPinnedMemory(test_tool_pack);
      CHECK_JUST(test_tool_pack->stream->stream()->Sync());

      ASSERT_TRUE(test_tensor_pack->VerifyResult());

      delete test_tensor_pack;
#if TEST_GATHER_LOG_OUTPUT
      std::cout << ">>>>>>>>>>>>>>> TEST PASSED <<<<<<<<<<<<<<<\n" << std::endl;
#endif
#undef TEST_GATHER_LOG_OUTPUT
    }

    delete test_tool_pack;
  }
}

}  // namespace

TEST_F(PrimitiveTest, TestGather) {
  TestGather<DataType::kInt32, int32_t, DataType::kInt32, int32_t>(&device_manager_registry_,
                                                                   available_device_types_);
}

}  // namespace test

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
