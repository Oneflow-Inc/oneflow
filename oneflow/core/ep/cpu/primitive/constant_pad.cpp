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

// todo 

template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template<typename T, size_t pack_size>
union Pack {
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
  explicit Pack(T value) {
#pragma unroll
    for (int i = 0; i < pack_size; i++) { elem[i] = value; }
  }
  T elem[pack_size];
  PackType<T, pack_size> storage;
};

template<size_t num_dims, typename IndexType, typename T, int pack_size>
void ConstantPadKernel(ConstantPadParams<num_dims, IndexType> params, T pad_value) {
  using LoadStoreType = PackType<T, pack_size>;
  const LoadStoreType* src = reinterpret_cast<const LoadStoreType*>(params.src);
  LoadStoreType* dst = reinterpret_cast<LoadStoreType*>(params.dst);
  IndexType src_index[num_dims];
  IndexType dst_index[num_dims];
  for (IndexType linear_index = 0; linear_index < params.elem_cnt; ++linear_index) {
    params.dst_index_helper.OffsetToNdIndex(linear_index, dst_index);
    bool if_pad = false;
#pragma unroll
    for (int i = 0; i < num_dims; i++) {
      if (dst_index[i] >= params.padding_before[i]
          && dst_index[i] < params.out_size[i] - params.padding_after[i]) {
        src_index[i] = dst_index[i] - params.padding_before[i];
      } else {
        if_pad = true;
        break;
      }
    }
    if (!if_pad) {
      const IndexType src_offset = params.src_index_helper.NdIndexToOffset(src_index);
      dst[linear_index] = src[src_offset];
    } else {
      Pack<T, pack_size> packed_pad_val(pad_value);
      dst[linear_index] = packed_pad_val.storage;
    }
  }
}

template<typename T>
T GetValue(Scalar value) {
  return value.Value<T>();
}

template<>
float16 GetValue<float16>(Scalar value) {
  return static_cast<float16>(GetValue<float>(value));
}

template<size_t num_dims, typename IndexType, typename T>
void LaunchKernel(Stream* stream, ConstantPadParams<num_dims, IndexType> params, T pad_val) {
  ConstantPadKernel<num_dims, IndexType, T, /*pack_size*/ 1>(params, pad_val);
}

template<size_t num_dims, typename IndexType, typename T>
void LaunchKernel(Stream* stream, void* dst, const int64_t* dst_dims, const void* src,
                  const int64_t* src_dims, const int64_t* padding_before,
                  const int64_t* padding_after, T pad_val) {
  ConstantPadParams<num_dims, IndexType> params;
  params.dst_index_helper = NdIndexOffsetHelperCalculator<IndexType, num_dims>(dst_dims);
  params.src_index_helper = NdIndexOffsetHelperCalculator<IndexType, num_dims>(src_dims);
  params.dst = dst;
  params.src = src;
  size_t elem_cnt = 1;
  for (int i = 0; i < num_dims; i++) {
    params.padding_before[i] = padding_before[i];
    params.padding_after[i] = padding_after[i];
    params.out_size[i] = dst_dims[i];
    elem_cnt *= params.out_size[i];
  }
  params.elem_cnt = elem_cnt;
  LaunchKernel<num_dims, IndexType, T>(stream, params, pad_val);
}

template<size_t num_dims, typename T>
void DispatchIndexType(Stream* stream, void* dst, const int64_t* dst_dims, const void* src,
                       const int64_t* src_dims, const int64_t* padding_before,
                       const int64_t* padding_after, T pad_val) {
  size_t elem_cnt = 1;
  for (size_t i = 0; i < num_dims; ++i) { elem_cnt *= dst_dims[i]; }
  if (elem_cnt < GetMaxVal<int32_t>()) {
    LaunchKernel<num_dims, int32_t, T>(stream, dst, dst_dims, src, src_dims, padding_before,
                                       padding_after, pad_val);
  } else {
    LaunchKernel<num_dims, int64_t, T>(stream, dst, dst_dims, src, src_dims, padding_before,
                                       padding_after, pad_val);
  }
}

template<typename T>
void LaunchWithSimplified(Stream* stream, size_t num_dims, void* dst, const int64_t* dst_dims,
                          const void* src, const int64_t* src_dims, const int64_t* padding_before,
                          const int64_t* padding_after, T pad_val) {
  void (*func)(Stream* /*stream*/, void* /*dst*/, const int64_t* /*dst_dims*/, const void* /*src*/,
               const int64_t* /*src_dims*/, const int64_t* /*padding_before*/,
               const int64_t* /*padding_after*/, T) = nullptr;
  if (num_dims == 1) {
    func = DispatchIndexType<1, T>;
  } else if (num_dims == 2) {
    func = DispatchIndexType<2, T>;
  } else if (num_dims == 3) {
    func = DispatchIndexType<3, T>;
  } else if (num_dims == 4) {
    func = DispatchIndexType<4, T>;
  } else if (num_dims == 5) {
    func = DispatchIndexType<5, T>;
  } else if (num_dims == 6) {
    func = DispatchIndexType<6, T>;
  } else if (num_dims == 7) {
    func = DispatchIndexType<7, T>;
  } else if (num_dims == 8) {
    func = DispatchIndexType<8, T>;
  } else {
    UNIMPLEMENTED();
  }
  func(stream, dst, dst_dims, src, src_dims, padding_before, padding_after, pad_val);
}

template<typename T>
class ConstantPadImpl : public ConstantPad {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantPadImpl);
  ConstantPadImpl() = default;
  ~ConstantPadImpl() override = default;

  void Launch(Stream* stream, size_t num_dims, void* dst, const int64_t* dst_dims, const void* src,
              const int64_t* src_dims, const int64_t* padding_before, const int64_t* padding_after,
              Scalar pad_val) override {
    LaunchWithSimplified<T>(stream, num_dims, dst, dst_dims, src, src_dims, padding_before,
                            padding_after, GetValue<T>(pad_val));
  }
};

template<typename T>
std::unique_ptr<ConstantPad> NewConstantPad() {
  return std::unique_ptr<ConstantPad>(new ConstantPadImpl<T>());
}

#define CPU_CONSTANT_PAD_PRIMITIVE_TYPE_SEQ \
  CPU_PRIMITIVE_INT32_TYPE_SEQ              \
  CPU_PRIMITIVE_INT64_TYPE_SEQ              \
  CPU_PRIMITIVE_FLOAT_TYPE_SEQ              \
  CPU_PRIMITIVE_DOUBLE_TYPE_SEQ

class ConstantPadFactoryImpl : public ConstantPadFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantPadFactoryImpl);
  ConstantPadFactoryImpl() = default;
  ~ConstantPadFactoryImpl() override = default;

  std::unique_ptr<ConstantPad> New(DataType data_type) override {
#define MAKE_NEW_CONSTANT_PAD_ENTRY(type_cpp, type_proto) {type_proto, NewConstantPad<type_cpp>},

    static const std::map<DataType, std::function<std::unique_ptr<ConstantPad>()>>
        new_constant_pad_handle{
            OF_PP_FOR_EACH_TUPLE(MAKE_NEW_CONSTANT_PAD_ENTRY, CPU_CONSTANT_PAD_PRIMITIVE_TYPE_SEQ)};

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
