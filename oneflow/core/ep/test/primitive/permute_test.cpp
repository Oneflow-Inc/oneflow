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
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include <Eigen/Core>
#include "stdio.h"
namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

void TestPermute(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types){
    const size_t m = 127; 
    const size_t n = 127; 
    const size_t elem_cnt = m*n; 
    const size_t matrix_size = elem_cnt * sizeof(float); 
    Eigen::MatrixXf mat = Eigen::MatrixXf::Random(m, n); 

    // for(int row = 0; row < m; row++){
    //     for(int col = 0; col < n; col++){
    //         printf("%f ", mat.data()[row*n + col]); 
    //     }
    //     printf("\n"); 
    // }

    for (const auto& device_type : device_types){
        auto device = registry->GetDevice(device_type, 0);
        AllocationOptions pinned_options;
        pinned_options.SetPinnedDevice(device_type, 0);
        AllocationOptions device_options;
        void* host_src; 
        void* host_dst; 
        void* device_src; 
        void* device_dst; 

        CHECK_JUST(device->AllocPinned(pinned_options, &host_src, matrix_size));
        CHECK_JUST(device->AllocPinned(pinned_options, &host_dst, matrix_size));
        CHECK_JUST(device->Alloc(device_options, &device_src, matrix_size));
        CHECK_JUST(device->Alloc(device_options, &device_dst, matrix_size));

        ep::test::StreamGuard stream(device.get());
        std::unique_ptr<Permute> permute = NewPrimitive<PermuteFactory>(device_type, /*max_num_dims=*/2);
        ASSERT_TRUE(permute.operator bool());
        std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
        std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
        ASSERT_TRUE(d2h.operator bool());
        ASSERT_TRUE(h2d.operator bool());
        std::memcpy(host_src, mat.data(), matrix_size); 
        h2d->Launch(stream.stream(), device_src, host_src, matrix_size); 
        // Launch(Stream* stream, DataType data_type, size_t num_dims, const int64_t* src_dims,
        //               const void* src, const int* permutation, void* dst)
        const int64_t src_dims[2] = {m, n};
        const int permutation[2] = {1, 0};
        permute->Launch(stream.stream(), DataType::kFloat, /*num_dims=*/2, src_dims, device_src, permutation, device_dst); 
        d2h->Launch(stream.stream(), host_dst, device_dst, matrix_size); 
        Eigen::MatrixXf mat_transposed = mat.transpose(); 

        // printf("=== Mat transpose === \n"); 
        // for(int col = 0; col < n; col++){
        //     for(int row = 0; row < m; row++){
        //         printf("%f ", mat_transposed.data()[col*m + row]); 
        //     }
        //     printf("\n"); 
        // }

        // printf("=== Primitive transpose === \n"); 
        // for(int col = 0; col < n; col++){
        //     for(int row = 0; row < m; row++){
        //         printf("%f ", reinterpret_cast<float*>(host_dst)[col*m + row]); 
        //     }
        //     printf("\n"); 
        // }
        // auto res = Eigen::Map<Eigen::MatrixXf, Eigen::Unaligned>(reinterpret_cast<float*>(host_dst), elem_cnt);
        // ASSERT_TRUE(mat_transposed.template isApprox(res));
        for(int i = 0; i < elem_cnt; i++){
            ASSERT_EQ(reinterpret_cast<float*>(mat_transposed.data())[i], reinterpret_cast<float*>(host_dst)[i]);
        }

        device->FreePinned(pinned_options, host_src);
        device->FreePinned(pinned_options, host_dst);
        device->Free(device_options, device_src);
        device->Free(device_options, device_dst);
    }
    
}

TEST_F(PrimitiveTest, TestBatchPermute){
    TestPermute(&device_manager_registry_, available_device_types_); 
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
