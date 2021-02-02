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
#ifndef ONEFLOW_CORE_COMMON_DATA_TYPE_CONVERTER_H_
#define ONEFLOW_CORE_COMMON_DATA_TYPE_CONVERTER_H_

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif
#include <cstdint>
#include <limits>
#include <type_traits>
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
struct IsFloatingOrHalf {
  static const bool value = IsFloating<T>::value || IsFloat16<T>::value;
};

template<typename T>
struct IsArithmeticOrHalf {
  static const bool value = std::is_arithmetic<T>::value || IsFloat16<T>::value;
};

template<typename From, typename To>
struct NeedsClamp {
  static const bool from_fp = IsFloatingOrHalf<From>::value;
  static const bool to_fp = IsFloatingOrHalf<To>::value;
  static const bool from_fp16 = IsFloat16<From>::value;
  static const bool to_fp16 = IsFloat16<To>::value;
  static const bool from_unsigned = std::is_unsigned<From>::value;
  static const bool to_unsigned = std::is_unsigned<To>::value;
  static const bool value =
      // to smaller type of same kind (fp, int)
      (from_fp == to_fp && sizeof(To) < sizeof(From)) ||
      // fp32 has range in excess of (u)int64
      (from_fp && !to_fp) ||
      // converting to unsigned requires clamping negatives to zero
      (!from_unsigned && to_unsigned) ||
      // zero-extending signed unsigned integers requires more bits
      (from_unsigned && !to_unsigned && sizeof(To) <= sizeof(From)) ||
      // float16
      (to_fp16 && sizeof(To) <= sizeof(From));
};

template<typename To>
struct NeedsClamp<bool, To> {
  static const bool value = false;
};

template<typename T, typename U, typename Enabled = void>
struct ClampHelper {};

// floating-point and signed integer -> floating-point and signed integer
template<typename T, typename U>
struct ClampHelper<
    T, U,
    std::enable_if_t<
        NeedsClamp<U, T>::value && std::is_signed<U>::value && std::is_signed<T>::value, void>> {
  OF_DEVICE_FUNC static const T Call(U value) {
    return value <= GetMinVal<T>()
               ? GetMinVal<T>()
               : value >= GetMaxVal<T>() ? GetMaxVal<T>() : static_cast<T>(value);
  }
};

// floating-point -> unsigned types
template<typename T, typename U>
struct ClampHelper<T, U,
                   std::enable_if_t<NeedsClamp<U, T>::value && std::is_signed<U>::value
                                        && IsFloatingOrHalf<U>::value && std::is_unsigned<T>::value,
                                    void>> {
  OF_DEVICE_FUNC static const T Call(U value) {
    return value <= GetMinVal<T>()
               ? GetMinVal<T>()
               : value >= GetMaxVal<T>() ? GetMaxVal<T>() : static_cast<T>(value);
  }
};

// signed integer types -> unsigned types
template<typename T, typename U>
struct ClampHelper<T, U,
                   std::enable_if_t<NeedsClamp<U, T>::value && std::is_signed<U>::value
                                        && std::is_integral<U>::value && std::is_unsigned<T>::value,
                                    void>> {
  OF_DEVICE_FUNC static const T Call(U value) {
    return value <= 0 ? 0
                      : static_cast<std::make_unsigned_t<U>>(value) >= GetMaxVal<T>()
                            ? GetMaxVal<T>()
                            : static_cast<T>(value);
  }
};

// unsigned types -> any types
template<typename T, typename U>
struct ClampHelper<T, U,
                   std::enable_if_t<NeedsClamp<U, T>::value && std::is_unsigned<U>::value, void>> {
  OF_DEVICE_FUNC static const T Call(U value) {
    return value >= GetMaxVal<T>() ? GetMaxVal<T>() : static_cast<T>(value);
  }
};

// not clamp
template<typename T, typename U>
struct ClampHelper<T, U, std::enable_if_t<!NeedsClamp<U, T>::value, void>> {
  OF_DEVICE_FUNC static const T Call(U value) { return value; }
};

OF_DEVICE_FUNC const int32_t Clamp(uint32_t value) {
  return value & 0x80000000u ? 0x7fffffff : value;
}

OF_DEVICE_FUNC const uint32_t Clamp(int32_t value) { return value < 0 ? 0u : value; }

OF_DEVICE_FUNC const int32_t Clamp(int64_t value) {
  return value < static_cast<int64_t>(GetMinVal<int32_t>())
             ? GetMinVal<int32_t>()
             : value > static_cast<int64_t>(GetMaxVal<int32_t>()) ? GetMaxVal<int32_t>()
                                                                  : static_cast<int32_t>(value);
}

template<>
struct ClampHelper<int32_t, uint64_t> {
  OF_DEVICE_FUNC static const int32_t Call(uint64_t value) {
    return value > static_cast<uint64_t>(GetMaxVal<int32_t>()) ? GetMaxVal<int32_t>()
                                                               : static_cast<int32_t>(value);
  }
};

template<>
struct ClampHelper<uint32_t, int64_t> {
  OF_DEVICE_FUNC static const uint32_t Call(int64_t value) {
    return value < 0
               ? 0
               : value > static_cast<int64_t>(GetMaxVal<uint32_t>()) ? GetMaxVal<uint32_t>()
                                                                     : static_cast<uint32_t>(value);
  }
};

template<>
struct ClampHelper<uint32_t, uint64_t> {
  OF_DEVICE_FUNC static const uint32_t Call(uint64_t value) {
    return value > static_cast<uint64_t>(GetMaxVal<uint32_t>()) ? GetMaxVal<uint32_t>()
                                                                : static_cast<uint32_t>(value);
  }
};

template<typename T>
struct ClampHelper<bool, T> {
  OF_DEVICE_FUNC static const bool Call(T value) { return static_cast<bool>(value); }
};

template<typename T>
struct ClampHelper<float16, T> {
  inline static const float16 Call(T value) {
    return static_cast<float16>(ClampHelper<T, float>::Call(value) < GetMinVal<float16>()
                                    ? GetMinVal<float16>()
                                    : ClampHelper<T, float>::Call(value) > GetMaxVal<float16>()
                                          ? GetMaxVal<float16>()
                                          : ClampHelper<T, float>::Call(value));
  }
};

template<typename T>
struct ClampHelper<T, float16> {
  inline static const T Call(float16 value) {
    return ClampHelper<T, float>::Call(static_cast<float>(value));
  }
};

inline const float16 Clamp(float16 value) { return value; }

template<typename T, typename U>
OF_DEVICE_FUNC const T Clamp(U value) {
  return ClampHelper<T, U>::Call(value);
}

namespace {
#ifdef __CUDA_ARCH__

inline __device__ int cuda_round_helper(float f, int) { return __float2int_rn(f); }

inline __device__ unsigned cuda_round_helper(float f, unsigned) { return __float2uint_rn(f); }

inline __device__ long long cuda_round_helper(float f, long long) {
  return __float2ll_rd(f + 0.5f);
}

inline __device__ unsigned long long cuda_round_helper(float f, unsigned long long) {
  return __float2ull_rd(f + 0.5f);
}

inline __device__ long cuda_round_helper(float f, long) {
  return sizeof(long) == sizeof(int) ? __float2int_rn(f) : __float2ll_rd(f + 0.5f);
}

inline __device__ unsigned long cuda_round_helper(float f, unsigned long) {
  return sizeof(unsigned long) == sizeof(unsigned int) ? __float2uint_rn(f)
                                                       : __float2ull_rd(f + 0.5f);
}

inline __device__ int cuda_round_helper(double f, int) { return __double2int_rn(f); }

inline __device__ unsigned cuda_round_helper(double f, unsigned) { return __double2uint_rn(f); }

inline __device__ long long cuda_round_helper(double f, long long) {
  return __double2ll_rd(f + 0.5f);
}

inline __device__ unsigned long long cuda_round_helper(double f, unsigned long long) {
  return __double2ull_rd(f + 0.5f);
}

inline __device__ long cuda_round_helper(double f, long) {
  return sizeof(long) == sizeof(int) ? __double2int_rn(f) : __double2ll_rd(f + 0.5f);
}

inline __device__ unsigned long cuda_round_helper(double f, unsigned long) {
  return sizeof(unsigned long) == sizeof(unsigned int) ? __double2uint_rn(f)
                                                       : __double2ull_rd(f + 0.5f);
}
#endif

template<typename Out, typename In, bool OutIsFp = IsFloatingOrHalf<Out>::value,
         bool InIsFp = IsFloatingOrHalf<In>::value>
struct ConverterBase;

template<typename Out, typename In>
struct Converter : ConverterBase<Out, In> {
  static_assert(IsArithmeticOrHalf<Out>::value && IsArithmeticOrHalf<In>::value,
                "Default ConverterBase can only be used with arithmetic types.");
};

// Converts between two FP types
template<typename Out, typename In>
struct ConverterBase<Out, In, true, true> {
  OF_DEVICE_FUNC static const Out Convert(In value) { return value; }
  OF_DEVICE_FUNC static const Out ConvertNorm(In value) { return value; }
  OF_DEVICE_FUNC static const Out ConvertSat(In value) { return value; }
  OF_DEVICE_FUNC static const Out ConvertSatNorm(In value) { return value; }
};

// Converts integral to FP type
template<typename Out, typename In>
struct ConverterBase<Out, In, true, false> {
  OF_DEVICE_FUNC static const Out Convert(In value) { return value; }
  OF_DEVICE_FUNC static const Out ConvertSat(In value) { return value; }
  OF_DEVICE_FUNC static const Out ConvertNorm(In value) {
    return value * (Out(1) / (GetMaxVal<In>()));
  }
  OF_DEVICE_FUNC static const Out ConvertSatNorm(In value) {
    return value * (Out(1) / (GetMaxVal<In>()));
  }
};

// Converts integral to float16
template<typename In>
struct ConverterBase<float16, In, true, false> {
  OF_DEVICE_FUNC static const float16 Convert(In value) {
    auto out = ConverterBase<float, In, true, false>::Convert(value);
    return static_cast<float16>(out);
  }

  OF_DEVICE_FUNC static const float16 ConvertSat(In value) {
    auto out = ConverterBase<float, In, true, false>::ConvertSat(value);
    return static_cast<float16>(out);
  }

  OF_DEVICE_FUNC static const float16 ConvertNorm(In value) {
    auto out = ConverterBase<float, In, true, false>::ConvertNorm(value);
    return static_cast<float16>(out);
  }

  OF_DEVICE_FUNC static const float16 ConvertSatNorm(In value) {
    auto out = ConverterBase<float, In, true, false>::ConvertSatNorm(value);
    return static_cast<float16>(out);
  }
};

// Converts FP to integral type
template<typename Out, typename In>
struct ConverterBase<Out, In, false, true> {
  OF_DEVICE_FUNC static const Out Convert(In value) {
#ifdef __CUDA_ARCH__
    return Clamp<Out>(cuda_round_helper(value, Out()));
#else
    return Clamp<Out>(std::round(value));
#endif
  }

  OF_DEVICE_FUNC static const Out ConvertSat(In value) {
#ifdef __CUDA_ARCH__
    return Clamp<Out>(cuda_round_helper(value, Out()));
#else
    return Clamp<Out>(std::round(value));
#endif
  }

  OF_DEVICE_FUNC static const Out ConvertNorm(In value) {
#ifdef __CUDA_ARCH__
    return Clamp<Out>(cuda_round_helper(value * GetMaxVal<Out>(), Out()));
#else
    return std::round(value * GetMaxVal<Out>());
#endif
  }

  OF_DEVICE_FUNC static const Out ConvertSatNorm(In value) {
#ifdef __CUDA_ARCH__
    return std::is_signed<Out>::value
               ? Clamp<Out>(cuda_round_helper(value * GetMaxVal<Out>(), Out()))
               : cuda_round_helper(GetMaxVal<Out>() * __saturatef(value), Out());
#else
    return Clamp<Out>(std::round(value * GetMaxVal<Out>()));
#endif
  }
};

// Converts signed to signed, unsigned to unsigned or unsigned to signed
template<typename Out, typename In, bool IsOutSigned = std::is_signed<Out>::value,
         bool IsInSigned = std::is_signed<In>::value>
struct ConvertIntInt {
  OF_DEVICE_FUNC static const Out Convert(In value) { return value; }
  OF_DEVICE_FUNC static const Out ConvertNorm(In value) {
    return Converter<Out, float>::Convert(value * (1.0f * GetMaxVal<Out>() / GetMaxVal<In>()));
  }
  OF_DEVICE_FUNC static const Out ConvertSat(In value) { return Clamp<Out>(value); }
  OF_DEVICE_FUNC static const Out ConvertSatNorm(In value) { return ConvertNorm(value); }
};

// Converts signed to unsigned integer
template<typename Out, typename In>
struct ConvertIntInt<Out, In, false, true> {
  OF_DEVICE_FUNC static const Out Convert(In value) { return value; }
  OF_DEVICE_FUNC static const Out ConvertNorm(In value) {
    return Converter<Out, float>::Convert(value * (1.0f * GetMaxVal<Out>() / GetMaxVal<In>()));
  }
  OF_DEVICE_FUNC static const Out ConvertSat(In value) { return Clamp<Out>(value); }
  OF_DEVICE_FUNC static const Out ConvertSatNorm(In value) {
#ifdef __CUDA_ARCH__
    return cuda_round_helper(__saturatef(value * (1.0f / GetMaxVal<In>())) * GetMaxVal<Out>());
#else
    return value < 0 ? 0 : ConvertNorm(value);
  }
#endif
  };

  // Converts between integral types
  template<typename Out, typename In>
  struct ConverterBase<Out, In, false, false> : ConvertIntInt<Out, In> {
    static_assert(IsArithmeticOrHalf<Out>::value && IsArithmeticOrHalf<In>::value,
                  "Default ConverterBase can only be used with arithmetic types.");
  };

  // Pass-through conversion
  template<typename T>
  struct Converter<T, T> {
    static OF_DEVICE_FUNC const T Convert(T value) { return value; }
    static OF_DEVICE_FUNC const T ConvertSat(T value) { return value; }
    static OF_DEVICE_FUNC const T ConvertNorm(T value) { return value; }
    static OF_DEVICE_FUNC const T ConvertSatNorm(T value) { return value; }
  };

  template<typename raw_out, typename raw_in>
  using converter_t =
      Converter<std::remove_cv_t<raw_out>, std::remove_cv_t<std::remove_reference_t<raw_in>>>;

}  // namespace

template<typename Out, typename In>
OF_DEVICE_FUNC const Out Convert(In value) {
  return converter_t<Out, In>::Convert(value);
}

template<typename Out, typename In>
OF_DEVICE_FUNC const Out ConvertNorm(In value) {
  return converter_t<Out, In>::ConvertNorm(value);
}

template<typename Out, typename In>
OF_DEVICE_FUNC const Out ConvertSat(In value) {
  return converter_t<Out, In>::ConvertSat(value);
}

template<typename Out, typename In>
OF_DEVICE_FUNC const Out ConvertSatNorm(In value) {
  return converter_t<Out, In>::ConvertSatNorm(value);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DATA_TYPE_CONVERTER_H_
