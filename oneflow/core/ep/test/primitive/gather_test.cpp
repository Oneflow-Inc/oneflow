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

template<typename ParamsType, typename IndicesType>
struct TestGatherTensorPack {
    using InOutTensor = Eigen::Tensor<ParamsType, 4, Eigen::RowMajor>;
    using IndicesTensor = Eigen::Tensor<IndicesType, 2, Eigen::RowMajor>;
    InOutTensor *params;
    IndicesTensor *indices;
    InOutTensor *out;
    TestGatherTensorPack(
        size_t batch_dim_size, 
        size_t outer_dim_size,
        size_t gather_dim_size,
        size_t inner_dim_size,
        size_t per_batch_indices_size
    ) : params(
            std::unique_ptr<InOutTensor>(new InOutTensor(batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size))
        ), 
        indices(std::unique_ptr<IndicesTensor>(new IndicesTensor(batch_dim_size, per_batch_indices_size))),
        out(std::unique_ptr<InOutTensor>(new InOutTensor(batch_dim_size, outer_dim_size, per_batch_indices_size, inner_dim_size))){
            params.setRandom();
            out.setZero();
            IndicesType* indices_data = indices.data();
            for(size_t i=0; i<batch_dim_size*per_batch_indices_size; i++){
                indices_data[i] = std::rand() % gather_dim_size;
            }
    }
};

template<typename ParamsType, typename IndicesType>
std::unique_ptr<TestGatherTensorPack<ParamsType,IndicesType>> NewTestGatherTensorPack(
        size_t batch_dim_size, size_t outer_dim_size, size_t gather_dim_size, size_t inner_dim_size, size_t per_batch_indices_size){
    return std::unique_ptr<TestGatherTensorPack<ParamsType,IndicesType>>(new TestGatherTensorPack<ParamsType,IndicesType>(batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size, per_batch_indices_size));
}

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


template<DataType params_type, typename ParamsType, DataType indices_type, typename IndicesType>
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
        for(const auto& e : test_shapes){
            std::cout << "bacth_dim_size:           " << e[0] << std::endl;
            std::cout << "outer_dim_size:           " << e[1] << std::endl;
            std::cout << "gather_dim_size:          " << e[2] << std::endl;
            std::cout << "inner_dim_size:           " << e[3] << std::endl;
            std::cout << "per_batch_indices_size:   " << e[4] << std::endl;
            auto test_tensor_pack 
                = NewTestGatherTensorPack<ParamsType, IndicesType>(e[0], e[1], e[2], e[3], e[4]);
        }
    }
} 

}  // namespace

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow