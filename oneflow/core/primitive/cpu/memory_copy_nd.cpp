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
#include "oneflow/core/primitive/include/memory_copy_nd.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace primitive {

namespace {

void Copy1D(void* dst, const void* src, size_t count) {
  LOG(ERROR) << "Copy1D ";
  std::memcpy(dst, src, count);
}

void Copy2D(void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width,
            size_t height) {
  LOG(ERROR) << "Copy2D ";
  unsigned char* dst_ptr = (unsigned char*)dst;
  const unsigned char* src_ptr = (unsigned char*)src;
  FOR_RANGE(size_t, i, 0, height) {
    Copy1D(dst_ptr, src_ptr, width);
    dst_ptr += dst_pitch;
    src_ptr += src_pitch;
  }
}

void Copy3D(size_t size_of_data_type, void* dst, const int64_t* dst_dims, const int64_t* dst_pos,
            const void* src, const int64_t* src_dims, const int64_t* src_pos,
            const int64_t* extent) {
  LOG(ERROR) << "Copy3D ";
  const size_t dst_pitch = dst_dims[2] * size_of_data_type;
  const size_t src_pitch = src_dims[2] * size_of_data_type;
  const size_t dst_inner_area = dst_dims[1] * dst_pitch;
  const size_t src_inner_area = src_dims[1] * src_pitch;
  const size_t width = extent[2] * size_of_data_type;
  const size_t height = extent[1];
  const size_t depth = extent[0];
  FOR_RANGE(size_t, i, 0, depth) {
    void* dst_2d = (unsigned char*)dst + (dst_pos[0] + i) * dst_inner_area + dst_pos[1] * dst_pitch
                   + dst_pos[2] * size_of_data_type;
    const void* src_2d = (unsigned char*)src + (src_pos[0] + i) * src_inner_area
                         + src_pos[1] * src_pitch + src_pos[2] * size_of_data_type;
    Copy2D(dst_2d, dst_pitch, src_2d, src_pitch, width, height);
  }
}

template<int32_t NDIMS>
void CopyNDCpuImpl(size_t size_of_data_type, void* dst, const int64_t* dst_dims,
                   const int64_t* dst_pos, const void* src, const int64_t* src_dims,
                   const int64_t* src_pos, const int64_t* extent) {
  NdIndexOffsetHelper<int64_t, NDIMS> src_helper(src_dims);
  NdIndexOffsetHelper<int64_t, NDIMS> dst_helper(dst_dims);
  NdIndexOffsetHelper<int64_t, NDIMS> copy_helper(extent);
  int64_t copy_elem_cnt = 1;
  FOR_RANGE(int64_t, i, 0, NDIMS) { copy_elem_cnt *= extent[i]; }
  FOR_RANGE(int64_t, i, 0, copy_elem_cnt) {
    int64_t copy_idx[NDIMS];
    int64_t src_idx[NDIMS];
    int64_t dst_idx[NDIMS];
    copy_helper.OffsetToNdIndex(i, copy_idx);
    FOR_RANGE(int64_t, j, 0, NDIMS) {
      src_idx[j] = src_pos[j] + copy_idx[j];
      dst_idx[j] = dst_pos[j] + copy_idx[j];
    }
    const int64_t src_offset = src_helper.NdIndexToOffset(src_idx);
    const int64_t dst_offset = dst_helper.NdIndexToOffset(dst_idx);
    unsigned char* dst_ptr = reinterpret_cast<unsigned char*>(dst) + dst_offset * size_of_data_type;
    const unsigned char* src_ptr =
        reinterpret_cast<const unsigned char*>(src) + src_offset * size_of_data_type;
    FOR_RANGE(int64_t, j, 0, size_of_data_type) { dst_ptr[j] = src_ptr[j]; }
  }
}

void CopyND(size_t size_of_data_type, size_t num_dims, void* dst, const int64_t* dst_dims,
            const int64_t* dst_pos, const void* src, const int64_t* src_dims,
            const int64_t* src_pos, const int64_t* extent) {
  LOG(ERROR) << "CopyND " << num_dims;
  if (num_dims == 4) {
    CopyNDCpuImpl<4>(size_of_data_type, dst, dst_dims, dst_pos, src, src_dims, src_pos, extent);
  } else if (num_dims == 5) {
    CopyNDCpuImpl<5>(size_of_data_type, dst, dst_dims, dst_pos, src, src_dims, src_pos, extent);
  } else if (num_dims == 6) {
    CopyNDCpuImpl<6>(size_of_data_type, dst, dst_dims, dst_pos, src, src_dims, src_pos, extent);
  } else {
    UNIMPLEMENTED();
  }
}

class MemoryCopyNdImpl : public MemoryCopyNd {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryCopyNdImpl);
  MemoryCopyNdImpl() = default;
  ~MemoryCopyNdImpl() = default;

  void Launch(StreamContext* stream_ctx, DataType data_type, size_t num_dims, void* dst,
              const int64_t* dst_dims, const int64_t* dst_pos, const void* src,
              const int64_t* src_dims, const int64_t* src_pos,
              const int64_t* extent) const override {
    const size_t size_of_data_type = GetSizeOfDataType(data_type);
    if (num_dims == 0) {
      Copy1D((unsigned char*)dst, (unsigned char*)src, size_of_data_type);
    } else if (num_dims == 1) {
      Copy1D((unsigned char*)dst + dst_pos[0] * size_of_data_type,
             (unsigned char*)src + src_pos[0] * size_of_data_type, extent[0] * size_of_data_type);
    } else if (num_dims == 2) {
      const size_t dst_pitch = dst_dims[1] * size_of_data_type;
      const size_t src_pitch = src_dims[1] * size_of_data_type;
      const size_t width = extent[1] * size_of_data_type;
      const size_t height = extent[0];
      void* dst_2d = (unsigned char*)dst + dst_pos[0] * dst_pitch + dst_pos[1] * size_of_data_type;
      const void* src_2d =
          (const unsigned char*)src + src_pos[0] * src_pitch + src_pos[1] * size_of_data_type;
      Copy2D(dst_2d, dst_pitch, src_2d, src_pitch, width, height);
    } else if (num_dims == 3) {
      Copy3D(size_of_data_type, dst, dst_dims, dst_pos, src, src_dims, src_pos, extent);
    } else {
      CopyND(size_of_data_type, num_dims, dst, dst_dims, dst_pos, src, src_dims, src_pos, extent);
    }
  }
};

class MemoryCopyNdFactoryImpl : public MemoryCopyNdFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryCopyNdFactoryImpl);
  MemoryCopyNdFactoryImpl() = default;
  ~MemoryCopyNdFactoryImpl() override = default;

  std::unique_ptr<MemoryCopyNd> New() override { return std::make_unique<MemoryCopyNdImpl>(); }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, MemoryCopyNdFactory, MemoryCopyNdFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
