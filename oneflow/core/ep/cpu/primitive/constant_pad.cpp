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
#include "oneflow/core/ep/include/primitive/constant_pad.h"
#include "oneflow/core/ep/common/primitive/constant_pad.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"

namespace oneflow {

namespace ep {

namespace primitive {

namespace {

template<size_t num_dims, typename IndexType, typename StorageType>
void ConstantPadKernel(ConstantPadParams<num_dims, IndexType> params, StorageType packed_pad_val) {
  const StorageType* src = reinterpret_cast<const StorageType*>(params.src);
  StorageType* dst = reinterpret_cast<StorageType*>(params.dst);
  IndexType src_index[num_dims];
  IndexType dst_index[num_dims];
  for (IndexType linear_index = 0; linear_index < params.elem_cnt; ++linear_index) {
    params.dst_index_helper.OffsetToNdIndex(linear_index, dst_index);
    bool if_pad = false;
    for (int i = 0; i < num_dims; i++) {
      if (dst_index[i] >= params.valid_start[i] && dst_index[i] < params.valid_end[i]) {
        src_index[i] = dst_index[i] - params.valid_start[i];
      } else {
        if_pad = true;
        break;
      }
    }
    StorageType dst_val = packed_pad_val;
    if (!if_pad) {
      const IndexType src_offset = params.src_index_helper.NdIndexToOffset(src_index);
      dst_val = src[src_offset];
    }
    dst[linear_index] = dst_val;
  }
}

template<>
float16 GetValue<float16>(Scalar value) {
  return static_cast<float16>(GetValue<float>(value));
}

template<>
bfloat16 GetValue<bfloat16>(Scalar value) {
  return static_cast<bfloat16>(GetValue<float>(value));
}

template<size_t num_dims, typename IndexType, typename StorageType>
void LaunchKernel(ConstantPadParams<num_dims, IndexType> params, StorageType packed_pad_val) {
  ConstantPadKernel<num_dims, IndexType, StorageType>(params, packed_pad_val);
}

template<size_t num_dims, typename IndexType, typename StorageType>
void LaunchKernel(void* dst, const int64_t* dst_dims, const void* src, const int64_t* src_dims,
                  const int64_t* padding_before, const int64_t* padding_after,
                  StorageType packed_pad_val, size_t elem_cnt) {
  ConstantPadParams<num_dims, IndexType> params;
  params.dst_index_helper = OffsetToIndexCalculator<IndexType, num_dims>(dst_dims);
  params.src_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(src_dims);
  params.dst = dst;
  params.src = src;
  for (int i = 0; i < num_dims; i++) {
    params.valid_start[i] = padding_before[i];
    params.valid_end[i] = dst_dims[i] - padding_after[i];
  }
  params.elem_cnt = elem_cnt;
  LaunchKernel<num_dims, IndexType, StorageType>(params, packed_pad_val);
}

template<size_t num_dims, typename StorageType>
void DispatchIndexType(void* dst, const int64_t* dst_dims, const void* src, const int64_t* src_dims,
                       const int64_t* padding_before, const int64_t* padding_after,
                       StorageType packed_pad_val, size_t elem_cnt) {
  if (elem_cnt < GetMaxVal<int32_t>()) {
    LaunchKernel<num_dims, int32_t, StorageType>(dst, dst_dims, src, src_dims, padding_before,
                                                 padding_after, packed_pad_val, elem_cnt);
  } else {
    LaunchKernel<num_dims, int64_t, StorageType>(dst, dst_dims, src, src_dims, padding_before,
                                                 padding_after, packed_pad_val, elem_cnt);
  }
}

template<size_t num_dims, typename T>
void DispatchPackSize(void* dst, int64_t* dst_dims, const void* src, int64_t* src_dims,
                      int64_t* padding_before, int64_t* padding_after, T pad_val) {
  constexpr int32_t max_packsize = GetMaxPackSize<T>();
  size_t launch_pack_size = GetLaunchPackSize<max_packsize>(num_dims, dst, dst_dims, src, src_dims,
                                                            padding_before, padding_after);

  dst_dims[num_dims - 1] /= launch_pack_size;
  src_dims[num_dims - 1] /= launch_pack_size;
  padding_before[num_dims - 1] /= launch_pack_size;
  padding_after[num_dims - 1] /= launch_pack_size;

  size_t elem_cnt = 1;
  for (int i = 0; i < num_dims; i++) { elem_cnt *= dst_dims[i]; }

  if (launch_pack_size == 1) {
    Pack<T, 1> packed_pad_val(pad_val);
    DispatchIndexType<num_dims, PackType<T, 1>>(dst, dst_dims, src, src_dims, padding_before,
                                                padding_after, packed_pad_val.storage, elem_cnt);
  } else if (launch_pack_size == 2) {
    Pack<T, 2> packed_pad_val(pad_val);
    DispatchIndexType<num_dims, PackType<T, 2>>(dst, dst_dims, src, src_dims, padding_before,
                                                padding_after, packed_pad_val.storage, elem_cnt);
  } else if (launch_pack_size == 4) {
    Pack<T, 4> packed_pad_val(pad_val);
    DispatchIndexType<num_dims, PackType<T, 4>>(dst, dst_dims, src, src_dims, padding_before,
                                                padding_after, packed_pad_val.storage, elem_cnt);
  } else if (launch_pack_size == 8) {
    Pack<T, 8> packed_pad_val(pad_val);
    DispatchIndexType<num_dims, PackType<T, 8>>(dst, dst_dims, src, src_dims, padding_before,
                                                padding_after, packed_pad_val.storage, elem_cnt);
  } else if (launch_pack_size == 16) {
    Pack<T, 16> packed_pad_val(pad_val);
    DispatchIndexType<num_dims, PackType<T, 16>>(dst, dst_dims, src, src_dims, padding_before,
                                                 padding_after, packed_pad_val.storage, elem_cnt);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void LaunchWithSimplified(size_t num_dims, void* dst, int64_t* dst_dims, const void* src,
                          int64_t* src_dims, int64_t* padding_before, int64_t* padding_after,
                          T pad_val) {
  void (*func)(void* /*dst*/, int64_t* /*dst_dims*/, const void* /*src*/, int64_t* /*src_dims*/,
               int64_t* /*padding_before*/, int64_t* /*padding_after*/, T) = nullptr;
  if (num_dims == 1) {
    func = DispatchPackSize<1, T>;
  } else if (num_dims == 2) {
    func = DispatchPackSize<2, T>;
  } else if (num_dims == 3) {
    func = DispatchPackSize<3, T>;
  } else if (num_dims == 4) {
    func = DispatchPackSize<4, T>;
  } else if (num_dims == 5) {
    func = DispatchPackSize<5, T>;
  } else if (num_dims == 6) {
    func = DispatchPackSize<6, T>;
  } else if (num_dims == 7) {
    func = DispatchPackSize<7, T>;
  } else if (num_dims == 8) {
    func = DispatchPackSize<8, T>;
  } else {
    UNIMPLEMENTED();
  }
  func(dst, dst_dims, src, src_dims, padding_before, padding_after, pad_val);
}

template<typename T>
void SimplifyThenLaunch(size_t num_dims, const int64_t* src_dims, const void* src,
                        const int64_t* padding_before, const int64_t* padding_after, T pad_val,
                        void* dst) {
  CHECK_GT(num_dims, 0) << "num_dims must greater than 0";
  CHECK_LE(num_dims, kMaxNumDims);
  int64_t simplified_dst_dims[kMaxNumDims];
  int64_t simplified_src_dims[kMaxNumDims];
  int64_t simplified_padding_before[kMaxNumDims];
  int64_t simplified_padding_after[kMaxNumDims];
  size_t simplified_num_dims = 1;
  SimplifyPadDims(num_dims, src_dims, padding_before, padding_after, &simplified_num_dims,
                  simplified_dst_dims, simplified_src_dims, simplified_padding_before,
                  simplified_padding_after);
  LaunchWithSimplified<T>(simplified_num_dims, dst, simplified_dst_dims, src, simplified_src_dims,
                          simplified_padding_before, simplified_padding_after, pad_val);
}

template<typename T>
class ConstantPadImpl : public ConstantPad {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantPadImpl);
  ConstantPadImpl() = default;
  ~ConstantPadImpl() override = default;

  void Launch(Stream* stream, size_t num_dims, const int64_t* src_dims, const void* src,
              const int64_t* padding_before, const int64_t* padding_after, Scalar pad_val,
              void* dst) override {
    SimplifyThenLaunch<T>(num_dims, src_dims, src, padding_before, padding_after,
                          GetValue<T>(pad_val), dst);
  }
};

template<typename T>
std::unique_ptr<ConstantPad> NewConstantPad() {
  return std::unique_ptr<ConstantPad>(new ConstantPadImpl<T>());
}

class ConstantPadFactoryImpl : public ConstantPadFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantPadFactoryImpl);
  ConstantPadFactoryImpl() = default;
  ~ConstantPadFactoryImpl() override = default;

  std::unique_ptr<ConstantPad> New(DataType data_type) override {
#define MAKE_NEW_CONSTANT_PAD_ENTRY(type_cpp, type_proto) {type_proto, NewConstantPad<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<ConstantPad>()>>
        new_constant_pad_handle{
            OF_PP_FOR_EACH_TUPLE(MAKE_NEW_CONSTANT_PAD_ENTRY, CPU_PRIMITIVE_ALL_TYPE_SEQ)};

#undef MAKE_NEW_CONSTANT_PAD_ENTRY

    const auto it = new_constant_pad_handle.find(data_type);
    if (it != new_constant_pad_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, ConstantPadFactory, ConstantPadFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
