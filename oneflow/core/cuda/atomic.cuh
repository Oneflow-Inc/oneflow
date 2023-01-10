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
#ifndef ONEFLOW_CORE_CUDA_ATOMIC_H_
#define ONEFLOW_CORE_CUDA_ATOMIC_H_

#if defined(__CUDACC__)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
namespace oneflow {

namespace cuda {

namespace atomic {

namespace internal {

template<typename T, typename U>
struct CastCASImpl {
  __device__ __forceinline__ T operator()(T* address, T compare, T val, bool* success) const {
    static_assert(sizeof(T) == sizeof(U), "");
    U assumed = *(reinterpret_cast<U*>(&compare));
    U ret = atomicCAS(reinterpret_cast<U*>(address), assumed, *(reinterpret_cast<U*>(&val)));
    *success = (ret == assumed);
    return *(reinterpret_cast<T*>(&ret));
  }
};

#if __CUDA_ARCH__ < 700 || (defined(__clang__) && defined(__CUDA__))

template<typename T>
struct CastCASImpl<T, unsigned short int> {
  __device__ __forceinline__ T operator()(T* address, T compare, T val, bool* success) const {
    static_assert(sizeof(T) == sizeof(unsigned short int), "");
    size_t offset = reinterpret_cast<size_t>(address) & 0x2;
    unsigned int* address_as_ui =
        reinterpret_cast<unsigned int*>(reinterpret_cast<char*>(address) - offset);
    unsigned int old = *address_as_ui;
    unsigned int assumed = *(reinterpret_cast<unsigned short int*>(&compare));
    unsigned int newval = *(reinterpret_cast<unsigned short int*>(&val));

    assumed = offset ? (old & 0xffff) | (assumed << 16) : (old & 0xffff0000) | assumed;
    newval = offset ? (old & 0xffff) | (newval << 16) : (old & 0xffff0000) | newval;

    unsigned int ret = atomicCAS(address_as_ui, assumed, newval);
    *success = (ret == assumed);
    ret = offset ? (ret >> 16) : (ret & 0xffff);
    return *(reinterpret_cast<T*>(&ret));
  }
};

#endif  // __CUDA_ARCH__

template<typename T>
__device__ __forceinline__ typename std::enable_if<sizeof(T) == sizeof(unsigned int), T>::type
CASImpl(T* address, T compare, T val, bool* success) {
  return CastCASImpl<T, unsigned int>()(address, compare, val, success);
}

template<typename T>
__device__ __forceinline__
    typename std::enable_if<sizeof(T) == sizeof(unsigned long long int), T>::type
    CASImpl(T* address, T compare, T val, bool* success) {
  return CastCASImpl<T, unsigned long long int>()(address, compare, val, success);
}

template<typename T>
__device__ __forceinline__ typename std::enable_if<sizeof(T) == sizeof(unsigned short int), T>::type
CASImpl(T* address, T compare, T val, bool* success) {
  return CastCASImpl<T, unsigned short int>()(address, compare, val, success);
}

__device__ __forceinline__ int CASImpl(int* address, int compare, int val, bool* success) {
  int ret = atomicCAS(address, compare, val);
  *success = (ret == compare);
  return ret;
}

__device__ __forceinline__ unsigned int CASImpl(unsigned int* address, unsigned int compare,
                                                unsigned int val, bool* success) {
  unsigned int ret = atomicCAS(address, compare, val);
  *success = (ret == compare);
  return ret;
}

__device__ __forceinline__ unsigned long long int CASImpl(unsigned long long int* address,
                                                          unsigned long long int compare,
                                                          unsigned long long int val,
                                                          bool* success) {
  unsigned long long int ret = atomicCAS(address, compare, val);
  *success = (ret == compare);
  return ret;
}

#if __CUDA_ARCH__ >= 700

__device__ __forceinline__ unsigned short int CASImpl(unsigned short int* address,
                                                      unsigned short int compare,
                                                      unsigned short int val, bool* success) {
  unsigned short int ret = atomicCAS(address, compare, val);
  *success = (ret == compare);
  return ret;
}

#endif  // __CUDA_ARCH__ >= 700

template<typename T>
struct AddOp {
  __device__ __forceinline__ T operator()(T a, T b) { return a + b; }
};

template<typename T, template<typename> class BinaryOp>
__device__ __forceinline__ T AtomicCASBinaryImpl(T* address, T val) {
  T old = *address;
  T assumed;
  bool success = false;
  do {
    assumed = old;
    old = CASImpl(address, assumed, BinaryOp<T>()(old, val), &success);
  } while (!success);
  return old;
}

template<typename T>
__device__ __forceinline__ T AddImpl(T* address, T val) {
  return AtomicCASBinaryImpl<T, AddOp>(address, val);
}

__device__ __forceinline__ int AddImpl(int* address, int val) { return atomicAdd(address, val); }

__device__ __forceinline__ unsigned int AddImpl(unsigned int* address, unsigned int val) {
  return atomicAdd(address, val);
}

__device__ __forceinline__ unsigned long long int AddImpl(unsigned long long int* address,
                                                          unsigned long long int val) {
  return atomicAdd(address, val);
}

__device__ __forceinline__ uint64_t AddImpl(uint64_t* address, uint64_t val) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long int), "");
  return static_cast<uint64_t>(atomicAdd(reinterpret_cast<unsigned long long int*>(address),
                                         static_cast<unsigned long long int>(val)));
}

__device__ __forceinline__ float AddImpl(float* address, float val) {
  return atomicAdd(address, val);
}

#if __CUDA_ARCH__ >= 600

__device__ __forceinline__ double AddImpl(double* address, double val) {
  return atomicAdd(address, val);
}

__device__ __forceinline__ half2 AddImpl(half2* address, half2 val) {
  return atomicAdd(address, val);
}

#endif  // __CUDA_ARCH__ >= 600

#if __CUDA_ARCH__ >= 700

__device__ __forceinline__ half AddImpl(half* address, half val) { return atomicAdd(address, val); }

#endif  // __CUDA_ARCH__ >= 700

#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ nv_bfloat16 AddImpl(nv_bfloat16* address, nv_bfloat16 val) {
  return atomicAdd(address, val);
}

__device__ __forceinline__ nv_bfloat162 AddImpl(nv_bfloat162* address, nv_bfloat162 val) {
  return atomicAdd(address, val);
}

#endif  // __CUDA_ARCH__ >= 800

#if __CUDA_ARCH__ < 530

__device__ __forceinline__ half2 AddImpl(half2* address, half2 val) {
  __trap();
  return val;
}

#endif  // __CUDA_ARCH__ < 530

}  // namespace internal

template<typename T, typename U>
__device__ __forceinline__ typename std::enable_if<!std::is_same<T, U>::value, T>::type Cast(U v) {
  return static_cast<T>(v);
}

template<typename T, typename U>
__device__ __forceinline__ typename std::enable_if<std::is_same<T, U>::value, T>::type Cast(U v) {
  return v;
}

template<typename T, typename U, typename V>
__device__ __forceinline__ T CAS(T* address, U compare, V val) {
  bool success = false;
  return internal::CASImpl(address, Cast<T>(compare), Cast<T>(val), &success);
}

template<typename T, typename U>
__device__ __forceinline__ T Add(T* address, U val) {
  return internal::AddImpl(address, Cast<T>(val));
}

__device__ __forceinline__ float Mul(int32_t* address, const int32_t val) {
  int32_t old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, val * assumed);
  } while (assumed != old);
  return old;
}

__device__ __forceinline__ float Mul(uint32_t* address, const uint32_t val) {
  uint32_t old = *address, assumed;
  do {
    assumed = old;
    old = atomicCAS(address, assumed, val * assumed);
  } while (assumed != old);
  return old;
}

__device__ __forceinline__ float Mul(uint64_t* address, const uint64_t val) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long int), "");
  unsigned long long int old = *reinterpret_cast<unsigned long long int*>(address), assumed;
  do {
    assumed = old;
    old = atomicCAS(reinterpret_cast<unsigned long long int*>(address), assumed,
                    static_cast<unsigned long long int>(val) * assumed);
  } while (assumed != old);
  return old;
}

__device__ __forceinline__ float Mul(float* address, const float val) {
  int32_t* address_as_int = reinterpret_cast<int32_t*>(address);
  int32_t old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val * __int_as_float(assumed)));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ __forceinline__ float Mul(double* address, const double val) {
  unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val * __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ __forceinline__ float Max(float* address, const float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i;
  int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ __forceinline__ double Max(double* address, const double val) {
  unsigned long long int* address_as_i = (unsigned long long int*)address;
  unsigned long long int old = *address_as_i;
  unsigned long long int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

// FastAdd is referenced from
// https://github.com/pytorch/pytorch/blob/396c3b1d88d7624938a2bb0b287f2a19f1e89bb4/aten/src/ATen/native/cuda/KernelUtils.cuh#L29
#if defined(__CUDACC__)
template<typename T, typename std::enable_if<std::is_same<half, T>::value>::type* = nullptr>
__device__ __forceinline__ void FastSpecializedAtomicAdd(T* base, size_t offset,
                                                         const size_t length, T value) {
#if ((defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) \
     || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  cuda::atomic::Add(reinterpret_cast<half*>(base) + offset, static_cast<half>(value));
#else
  // Accounts for the chance base falls on an odd 16 bit alignment (ie, not 32 bit aligned)
  __half* target_addr = reinterpret_cast<__half*>(base + offset);
  bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (low_byte && offset < (length - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __float2half_rz(0);
    cuda::atomic::Add(reinterpret_cast<__half2*>(target_addr), value2);

  } else if (!low_byte && offset > 0) {
    __half2 value2;
    value2.x = __float2half_rz(0);
    value2.y = value;
    cuda::atomic::Add(reinterpret_cast<__half2*>(target_addr - 1), value2);

  } else {
    cuda::atomic::Add(reinterpret_cast<__half*>(base) + offset, static_cast<__half>(value));
  }
#endif
}

template<typename T, typename std::enable_if<!std::is_same<half, T>::value>::type* = nullptr>
__device__ __forceinline__ void FastSpecializedAtomicAdd(T* base, size_t offset,
                                                         const size_t length, T value) {
  cuda::atomic::Add(base + offset, value);
}

template<class T>
__device__ __forceinline__ void FastAdd(T* base, size_t offset, const size_t length, T value) {
  FastSpecializedAtomicAdd(base, offset, length, value);
}
#endif

}  // namespace atomic

}  // namespace cuda

}  // namespace oneflow

#endif  // defined(__CUDACC__)

#endif  // ONEFLOW_CORE_CUDA_ATOMIC_H_
