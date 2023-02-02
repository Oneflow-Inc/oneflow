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
    StreamGuard *stream;

    TestGatherToolPack(DeviceType device_type, std::shared_ptr<Device> &device){
        h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
        d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
        assert(d2h.operator bool());
        assert(h2d.operator bool());
        stream = new StreamGuard(device.get());
    }

    ~TestGatherToolPack(){
        delete stream;
    }
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
    size_t per_batch_indices_size;
    size_t params_size;
    size_t indices_size;
    size_t out_size;

    InOutTensor *params;
    IndicesTensor *indices;
    InOutTensor *out;

    PinnedMemoryGuard *host_params;
    PinnedMemoryGuard *host_indices;
    PinnedMemoryGuard *host_out;
    DeviceMemoryGuard *device_params;
    DeviceMemoryGuard *device_indices;
    DeviceMemoryGuard *device_out;

    TestGatherTensorPack(
        size_t batch_dim_size, 
        size_t outer_dim_size,
        size_t gather_dim_size,
        size_t inner_dim_size,
        size_t per_batch_indices_size
    ) : batch_dim_size(batch_dim_size), 
        outer_dim_size(outer_dim_size),
        gather_dim_size(gather_dim_size),
        inner_dim_size(inner_dim_size),
        per_batch_indices_size(per_batch_indices_size){
            params = new InOutTensor(batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size);
            indices = new IndicesTensor(batch_dim_size, per_batch_indices_size);
            out = new InOutTensor(batch_dim_size, outer_dim_size, per_batch_indices_size, inner_dim_size);

            params->setRandom();
            out->setZero();
            IndicesType* indices_data = indices->data();
            for(size_t i=0; i<batch_dim_size*per_batch_indices_size; i++){
                indices_data[i] = std::rand() % gather_dim_size;
            }
    }

    ~TestGatherTensorPack() {
        delete params;
        delete indices;
        delete out;
        if(host_params != nullptr) delete host_params;
        if(host_indices != nullptr) delete host_indices;
        if(host_out != nullptr) delete host_out;
        if(device_params != nullptr) delete device_params;
        if(device_indices != nullptr) delete device_indices;
        if(device_out != nullptr) delete device_out;
    }

    void AllocateMemory(std::shared_ptr<Device> &device){
        params_size = batch_dim_size * outer_dim_size * gather_dim_size * inner_dim_size;
        indices_size = batch_dim_size * per_batch_indices_size;
        out_size = batch_dim_size * outer_dim_size * per_batch_indices_size * inner_dim_size;

        host_params = new PinnedMemoryGuard(device.get(), params_size);
        host_indices = new PinnedMemoryGuard(device.get(), indices_size);
        host_out = new PinnedMemoryGuard(device.get(), out_size);
        device_params = new DeviceMemoryGuard(device.get(), params_size);
        device_indices = new DeviceMemoryGuard(device.get(), indices_size);
        device_out = new DeviceMemoryGuard(device.get(), out_size);
    }

    void CopySrcDataToPinnedMemory(std::shared_ptr<Device> &device){
        std::memcpy(host_params->ptr(), params->data(), params_size);
        std::memcpy(host_indices->ptr(), indices->data(), indices_size);
    }

    void CopySrcDataToDevice(TestGatherToolPack *tool){
        tool->h2d->Launch(tool->stream->stream(), device_params->ptr<ParamsType>(), host_params->ptr<ParamsType>(), params_size);
        tool->h2d->Launch(tool->stream->stream(), device_indices->ptr<ParamsType>(), host_indices->ptr<ParamsType>(), indices_size);
    }

    void CopyDstDataToPinnedMemory(TestGatherToolPack *tool){
        tool->d2h->Launch(tool->stream->stream(), host_out->ptr<ParamsType>(), device_out->ptr<ParamsType>, out_size);
    }
};

template<typename T>
std::vector<std::vector<T>> CartProduct (const std::vector<std::vector<T>>& v) {
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


template<typename ParamsType, typename IndicesType>
void TestGather(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
    std::vector<std::vector<int>> test_set = {
        {16, 128, 1024},    // batch_dim_size
        {16, 128, 1024},    // outer_dim_size
        {16, 128, 1024},    // gather_dim_size
        {16, 128, 1024},    // inner_dim_size
        {32, 64, 128}       // per_batch_indices_size
    };

    std::vector<std::vector<int>> test_shapes = CartProduct<int>(test_set);

    for (const auto& device_type : device_types) {
        auto device = registry->GetDevice(device_type, 0);

        TestGatherToolPack *test_tool_pack = new TestGatherToolPack(device_type, device);

        for(const auto& e : test_shapes){
            std::cout << "bacth_dim_size:           " << e[0] << std::endl;
            std::cout << "outer_dim_size:           " << e[1] << std::endl;
            std::cout << "gather_dim_size:          " << e[2] << std::endl;
            std::cout << "inner_dim_size:           " << e[3] << std::endl;
            std::cout << "per_batch_indices_size:   " << e[4] << std::endl;
            TestGatherTensorPack<ParamsType, IndicesType> *test_tensor_pack = 
                new TestGatherTensorPack<ParamsType, IndicesType>(e[0], e[1], e[2], e[3], e[4]);

            test_tensor_pack->AllocateMemory(device);
            test_tensor_pack->CopySrcDataToPinnedMemory(device);
            test_tensor_pack->CopySrcDataToDevice(test_tool_pack);


            delete test_tensor_pack;

        }
    }
}

}  // namespace

TEST_F(PrimitiveTest, TestGather) {
    TestGather<int32_t, int32_t>(&device_manager_registry_, available_device_types_);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow