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
__device__ __forceinline__ T CastCASImpl(T* address, T compare, T val) {
  static_assert(sizeof(T) == sizeof(U), "");
  U ret = atomicCAS(reinterpret_cast<U*>(address), *(reinterpret_cast<U*>(&compare)),
                    *(reinterpret_cast<U*>(&val)));
  return *(reinterpret_cast<T*>(&ret));
}

template<typename T>
__device__ __forceinline__ typename std::enable_if<sizeof(T) == sizeof(unsigned int), T>::type
CASImpl(T* address, T compare, T val) {
  return CastCASImpl<T, unsigned int>(address, compare, val);
}

template<typename T>
__device__ __forceinline__
    typename std::enable_if<sizeof(T) == sizeof(unsigned long long int), T>::type
    CASImpl(T* address, T compare, T val) {
  return CastCASImpl<T, unsigned long long int>(address, compare, val);
}

template<typename T>
__device__ __forceinline__ typename std::enable_if<sizeof(T) == sizeof(unsigned short int), T>::type
CASImpl(T* address, T compare, T val) {
#if __CUDA_ARCH__ >= 700
  return CastCASImpl<T, unsigned short int>(address, compare, val);
#else
  __trap();
  return 0;
#endif  // __CUDA_ARCH__ >= 700
}

__device__ __forceinline__ int CASImpl(int* address, int compare, int val) {
  return atomicCAS(address, compare, val);
}

__device__ __forceinline__ unsigned int CASImpl(unsigned int* address, unsigned int compare,
                                                unsigned int val) {
  return atomicCAS(address, compare, val);
}

__device__ __forceinline__ unsigned long long int CASImpl(unsigned long long int* address,
                                                          unsigned long long int compare,
                                                          unsigned long long int val) {
  return atomicCAS(address, compare, val);
}

#if __CUDA_ARCH__ >= 700

__device__ __forceinline__ unsigned short int CASImpl(unsigned short int* address,
                                                      unsigned short int compare,
                                                      unsigned short int val) {
  return atomicCAS(address, compare, val);
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
  do {
    assumed = old;
    old = CASImpl(address, assumed, BinaryOp<T>()(old, val));
  } while (old != assumed);
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
  return internal::CASImpl(address, Cast<T>(compare), Cast<T>(val));
}

template<typename T, typename U>
__device__ __forceinline__ T Add(T* address, U val) {
  return internal::AddImpl(address, Cast<T>(val));
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

}  // namespace atomic

}  // namespace cuda

}  // namespace oneflow

#endif  // defined(__CUDACC__)

#endif  // ONEFLOW_CORE_CUDA_ATOMIC_H_
