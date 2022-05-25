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
#ifndef ONEFLOW_CORE_CUDA_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_CUDA_RANDOM_GENERATOR_H_

namespace oneflow {

namespace cuda {

namespace random {

namespace internal { 

#define ONE_PHILOX_W32_0   (0x9E3779B9)
#define ONE_PHILOX_W32_1   (0xBB67AE85)
#define ONE_PHILOX_M4x32_0 (0xD2511F53)
#define ONE_PHILOX_M4x32_1 (0xCD9E8D57)

struct curandStatePhilox4_32_10 {
    uint4 ctr;
    // uint4 output;
    uint2 key;
    // unsigned int STATE;
    // int boxmuller_flag;
    // int boxmuller_flag_double;
    // float boxmuller_extra;
    // double boxmuller_extra_double;
};


__forceinline__ __device__ void Philox_State_Incr(internal::curandStatePhilox4_32_10* s, unsigned long long n)
{
   unsigned int nlo = (unsigned int)(n);
   unsigned int nhi = (unsigned int)(n>>32);

   s->ctr.x += nlo;
   if( s->ctr.x < nlo )
      nhi++;

   s->ctr.y += nhi;
   if(nhi <= s->ctr.y)
      return;
   if(++s->ctr.z) return;
   ++s->ctr.w;
}

__forceinline__ __device__ void Philox_State_Incr_hi(internal::curandStatePhilox4_32_10* s, unsigned long long n)
{
   unsigned int nlo = (unsigned int)(n);
   unsigned int nhi = (unsigned int)(n>>32);

   s->ctr.z += nlo;
   if( s->ctr.z < nlo )
      nhi++;

   s->ctr.w += nhi;
}

__forceinline__ __device__ void Philox_State_Incr(internal::curandStatePhilox4_32_10* s)
{
   if(++s->ctr.x) return;
   if(++s->ctr.y) return;
   if(++s->ctr.z) return;
   ++s->ctr.w;
}

__forceinline__ __device__ unsigned int mulhilo32(unsigned int a, unsigned int b, unsigned int* hip)
{
#ifndef __CUDA_ARCH__
   // host code
   unsigned long long product = ((unsigned long long)a) * ((unsigned long long)b);
   *hip = product >> 32;
   return (unsigned int)product;
#else
   // device code
   *hip = __umulhi(a,b);
   return a*b;
#endif
}

__forceinline__ __device__ uint4 philox4x32round(uint4 ctr, uint2 key)
{
   unsigned int hi0;
   unsigned int hi1;
   unsigned int lo0 = internal::mulhilo32(ONE_PHILOX_M4x32_0, ctr.x, &hi0);
   unsigned int lo1 = internal::mulhilo32(ONE_PHILOX_M4x32_1, ctr.z, &hi1);

   uint4 ret  = {hi1^ctr.y^key.x, lo1, hi0^ctr.w^key.y, lo0};
   return ret;
}

__forceinline__ __device__ uint4 curand_Philox4x32_10(uint4 c, uint2 k)
{
   c = internal::philox4x32round(c, k);                           // 1 
   k.x += ONE_PHILOX_W32_0; k.y += ONE_PHILOX_W32_1;
   c = internal::philox4x32round(c, k);                           // 2
   k.x += ONE_PHILOX_W32_0; k.y += ONE_PHILOX_W32_1;
   c = internal::philox4x32round(c, k);                           // 3 
   k.x += ONE_PHILOX_W32_0; k.y += ONE_PHILOX_W32_1;
   c = internal::philox4x32round(c, k);                           // 4 
   k.x += ONE_PHILOX_W32_0; k.y += ONE_PHILOX_W32_1;
   c = internal::philox4x32round(c, k);                           // 5 
   k.x += ONE_PHILOX_W32_0; k.y += ONE_PHILOX_W32_1;
   c = internal::philox4x32round(c, k);                           // 6 
   k.x += ONE_PHILOX_W32_0; k.y += ONE_PHILOX_W32_1;
   c = internal::philox4x32round(c, k);                           // 7 
   k.x += ONE_PHILOX_W32_0; k.y += ONE_PHILOX_W32_1;
   c = internal::philox4x32round(c, k);                           // 8 
   k.x += ONE_PHILOX_W32_0; k.y += ONE_PHILOX_W32_1;
   c = internal::philox4x32round(c, k);                           // 9 
   k.x += ONE_PHILOX_W32_0; k.y += ONE_PHILOX_W32_1;
   return internal::philox4x32round(c, k);                        // 10
}

__forceinline__ __device__ uint4 curand4(internal::curandStatePhilox4_32_10 *state)
{
    // uint4 r;
    // uint4 tmp = state->output;
    // internal::Philox_State_Incr(state);
    // state->output= internal::curand_Philox4x32_10(state->ctr,state->key);
    // switch(state->STATE){
    // case 0:
    //     return tmp;
    // case 1:
    //     r.x = tmp.y;
    //     r.y = tmp.z;
    //     r.z = tmp.w;
    //     r.w = state->output.x;
    //     break;
    // case 2:
    //     r.x = tmp.z;
    //     r.y = tmp.w;
    //     r.z = state->output.x;
    //     r.w = state->output.y;
    //     break;
    // case 3:
    //     r.x = tmp.w;
    //     r.y = state->output.x;
    //     r.z = state->output.y;
    //     r.w = state->output.z;
    //     break;
    // default:
    //     // NOT possible but needed to avoid compiler warnings
    //     return tmp;
    // }
    // return r;

    internal::Philox_State_Incr(state);
    // state->output = internal::curand_Philox4x32_10(state->ctr,state->key);
    // return state->output; 
    return internal::curand_Philox4x32_10(state->ctr,state->key);
}

__forceinline__ __device__ void skipahead(unsigned long long n, internal::curandStatePhilox4_32_10 *state)
{
    // state->STATE += (n & 3);
    // n /= 4;
    // if( state->STATE > 3 ){
    //     n += 1;
    //     state->STATE -= 4;
    // }
    internal::Philox_State_Incr(state, n);
    // state->output = internal::curand_Philox4x32_10(state->ctr,state->key);
}

__forceinline__ __device__ void skipahead_sequence(unsigned long long n, internal::curandStatePhilox4_32_10 *state)
{
    internal::Philox_State_Incr_hi(state, n);
    // state->output = internal::curand_Philox4x32_10(state->ctr,state->key);
}


// __forceinline__ __device__ void curand_init(unsigned long long seed,
//     unsigned long long subsequence,
//     unsigned long long offset,
//     curandStatePhilox4_32_10 *state)
// {
//     state->ctr = make_uint4(0, 0, 0, 0);
//     state->key.x = (unsigned int)seed;
//     state->key.y = (unsigned int)(seed>>32);
//     // state->STATE = 0;
//     // skipahead_sequence(subsequence, state);
//     skipahead(offset, state);
// }

__forceinline__ __device__ void curand_init(unsigned long long seed,
    // unsigned long long subsequence,
    unsigned long long offset,
    internal::curandStatePhilox4_32_10 *state)
{
    state->ctr = make_uint4(0, 0, 0, 0);
    state->key.x = (unsigned int)seed;
    state->key.y = (unsigned int)(seed>>32);
    // state->STATE = 0;
    // skipahead_sequence(subsequence, state);
    internal::skipahead(offset, state);
}

}  // namespace internal

}  // namespace random

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_RANDOM_GENERATOR_H_
