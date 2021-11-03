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
#ifndef ONEFLOW_USER_KERNELS_RNNT_KERNEL_GPU_H_
#define ONEFLOW_USER_KERNELS_RNNT_KERNEL_GPU_H_

#include <tuple>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "oneflow/user/kernels/rnnt_kernel_helper.h"
#include "oneflow/user/kernels/rnnt_kernel_util.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

const int warp_size = 32;

template<int NT, typename T, typename Rop>
struct CTAReduce;

template<int NT, typename T, typename Rop>
struct CTAReduce {
    enum { Size = NT, Capacity = NT };
    struct Storage { T shared[Capacity]; };

    __device__ static T reduce(int tid, T x, Storage& storage, int count, Rop g) {
        T* s = storage.shared;
        s[tid] = x;
        __syncthreads();

#pragma unroll
        for(int offset = NT / 2; offset >= warp_size; offset /= 2) {
            if(tid + offset < count && tid < offset) {
                x = g(x, s[offset + tid]);
                s[tid] = x;
            }
            __syncthreads();
        }

        T shuff;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
#if CUDA_VERSION < 9000
            shuff = __shfl_down(x, offset);
#else
            shuff = __shfl_down_sync(0xFFFFFFFF, x, offset);
#endif
            if (tid + offset < count && tid < offset)
                x = g(x, shuff);
        }
        return x;
    }
};

template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_rows(Iop f, Rop g, const T*  acts, T* output, int num_rows) {

    typedef CTAReduce<NT, T, Rop> R;
    __shared__ typename R::Storage storage;

    int tid = threadIdx.x;
    int idx = tid;
    int col = blockIdx.x;
    T curr;

    if (idx < num_rows) {
        curr = f(acts[col * num_rows + idx]);
    }
    idx += NT;

    while (idx < num_rows) {
        curr = g(curr, f(acts[col * num_rows + idx]));
        idx += NT;
    }

    curr = R::reduce(tid, curr, storage, num_rows, g);

    if (tid == 0)
        output[col] = curr;
}

template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_minus(Iop f, Rop g, const T*  acts, T* output, int num_rows) {

    typedef CTAReduce<NT, T, Rop> R;
    __shared__ typename R::Storage storage;

    int tid = threadIdx.x;
    int idx = tid;
    int col = blockIdx.x;
    T curr;
    T max = output[col];

    if (idx < num_rows) {
        curr = f(acts[col * num_rows + idx] - max);
    }
    idx += NT;

    while (idx < num_rows) {
        curr = g(curr, f(acts[col * num_rows + idx] - max));
        idx += NT;
    }

    curr = R::reduce(tid, curr, storage, num_rows, g);

    if (tid == 0)
        output[col] = -max - log(curr);
}

struct ReduceHelper {

    template<typename T, typename Iof, typename Rof>
    static void impl(Iof f, Rof g, const T*  acts, T* output, int num_rows, int num_cols, bool minus, cudaStream_t stream) {

        int grid_size;

        if (minus) {
            grid_size = num_cols;
            reduce_minus<128><<<grid_size, 128, 0, stream>>>
               (f, g, acts, output, num_rows);

        } else {
            grid_size = num_cols;
            reduce_rows<128><<<grid_size, 128, 0, stream>>>
               (f, g, acts, output, num_rows);
        }
    }
};


template<typename T, typename Iof, typename  Rof>
rnntStatus_t reduce(Iof f, Rof g, const T*  acts, T* output, int rows, int cols, bool minus, cudaStream_t stream) {
    ReduceHelper::impl(f, g, acts, output, rows, cols, minus, stream);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return RNNT_STATUS_EXECUTION_FAILED;

    return RNNT_STATUS_SUCCESS;
}

template<typename T>
rnntStatus_t reduce_exp(const T*  acts, T *denom, int rows, int cols, bool minus, cudaStream_t stream) {
    return reduce(rnnt_helper::exponential<T>(), rnnt_helper::add<T>(), acts, denom, rows, cols, minus, stream);
}

template<typename T>
rnntStatus_t reduce_max(const T*  acts, T *denom, int rows, int cols, bool minus, cudaStream_t stream) {
    return reduce(rnnt_helper::identity<T>(), rnnt_helper::maximum<T>(), acts, denom, rows, cols, minus, stream);
}


template<typename T>
inline __device__ T logp(const T*  denom, const T*  acts, const int maxT, const int maxU, const int alphabet_size, int mb, int t, int u, int v) {
    const int col = (mb * maxT + t) * maxU + u;
    return denom[col] + acts[col * alphabet_size + v];
}


template<typename T>
__global__ void negp(T* in_buf, int32_t size) {
  CUDA_1D_KERNEL_LOOP(i, size) { in_buf[i] = -in_buf[i]; }
}


template<typename Tp>
__global__ void compute_alphas_kernel(const Tp*  acts, const Tp*  denom, Tp* alphas, Tp* llForward, const int*  xlen, const int*  ylen, 
    const int*  mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int b = blockIdx.x; 
    int u = threadIdx.x; 
    

    const int T = xlen[b];
    
    const int U = ylen[b] + 1;
    
    const int* labels = mlabels + b * (maxU - 1); 
    const int offset = b * maxT * maxU;
    alphas += offset;
    
    if (u == 0) alphas[0] = 0;
    
    __syncthreads();
    
    for (int n = 1; n < T+U-1; ++n) {
        int t = n - u;
        if (u == 0) {
            if (t > 0 && t < T) {
                alphas[t * maxU + u] = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, b, t-1, 0, blank_);
            }
        } else if (u < U) {
            if (t == 0)
                alphas[u] = alphas[u-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, 0, u-1, labels[u-1]);
            else if (t > 0 && t < T) {
                Tp no_emit = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, b, t-1, u, blank_);
                Tp emit = alphas[t * maxU + u-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u-1, labels[u-1]);
                alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
        __syncthreads();
    }

    if (u == 0) {
        Tp loglike = alphas[(T-1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, T-1, U-1, blank_);
        llForward[b] = loglike;
    }
}

template<typename Tp>
__global__ void compute_alphas_kernel_naive(const Tp*  acts, const Tp*  denom, Tp* alphas, Tp* llForward, const int*  xlen, const int*  ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; 
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1); 
    const int offset = tid * maxT * maxU;
    alphas += offset;
    alphas[0] = 0;

    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            if (u == 0 && t > 0)
                alphas[t * maxU + u] = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t-1, 0, blank_);
            if (t == 0 && u > 0)
                alphas[u] = alphas[u-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, 0, u-1, labels[u-1]);
            if (t > 0 && u > 0) {
                Tp no_emit = alphas[(t-1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t-1, u, blank_);
                Tp emit = alphas[t * maxU + u-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u-1, labels[u-1]);
                alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    Tp loglike = alphas[(T-1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, U-1, blank_);
    llForward[tid] = loglike;
}


template<typename Tp>
__global__ void compute_betas_kernel(const Tp*  acts, const Tp*  denom, Tp* betas, Tp* llBackward, const int*  xlen, const int*  ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int b = blockIdx.x; 
    int u = threadIdx.x; 
    const int T = xlen[b];
    const int U = ylen[b] + 1;
    const int* labels = mlabels + b * (maxU - 1);
    const int offset = b * maxT * maxU;
    betas += offset;
    if (u == 0)
        betas[(T-1) * maxU + U-1] = logp(denom, acts, maxT, maxU, alphabet_size, b, T-1, U-1, blank_);

    __syncthreads();
    for (int n = T+U-2; n >= 0; --n) {
        int t = n - u;
        if (u == U-1) {
            if (t >= 0 && t < T-1)
                betas[t * maxU + U-1] = betas[(t+1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, U-1, blank_);
        } else if (u < U) {
            if (t == T-1)
                betas[(T-1) * maxU + u] = betas[(T-1) * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, b, T-1, u, labels[u]);
            else if (t >= 0 && t < T-1) {
                Tp no_emit = betas[(t+1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_);
                Tp emit = betas[t * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u]);
                betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
        __syncthreads();
    }

    if (u == 0) {
        llBackward[b] = betas[0];
    }
}

template<typename Tp>
__global__ void compute_betas_kernel_naive(const Tp*  acts, const Tp*  denom, Tp* betas, Tp* llBackward, const int*  xlen, const int*  ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1);
    const int offset = tid * maxT * maxU;
    betas += offset;
    betas[(T-1) * maxU + U-1] = logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, U-1, blank_);

    for (int t = T-1; t >=0; --t) {
        for (int u = U-1; u >= 0; --u) {
            if (u == U-1 && t < T-1)
                betas[t * maxU + U-1] = betas[(t+1) * maxU + U-1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, U-1, blank_);
            if (t == T-1 && u < U-1)
                betas[(T-1) * maxU + u] = betas[(T-1) * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, T-1, u, labels[u]);
            if (t < T-1 && u < U-1) {
                Tp no_emit = betas[(t+1) * maxU + u] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u, blank_);
                Tp emit = betas[t * maxU + u+1] + logp(denom, acts, maxT, maxU, alphabet_size, tid, t, u, labels[u]);
                betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    llBackward[tid] = betas[0];
}

template<int NT, typename Tp>
__global__ void compute_grad_kernel(Tp* grads, const Tp*  acts, const Tp*  denom, const Tp* alphas, const Tp* betas, const Tp*  logll, const int*  xlen, const int*  ylen, 
    const int*  mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_) {
    int tid = threadIdx.x; 
    int idx = tid;
    int col = blockIdx.x; 

    int u = col % maxU;
    int bt = (col - u) / maxU;
    int t = bt % maxT;
    int mb = (bt - t) / maxT;

    const int T = xlen[mb];
    const int U = ylen[mb] + 1;
    const int* labels = mlabels + mb * (maxU - 1);

    if (t < T && u < U) {
        while (idx < alphabet_size) {
            Tp logpk = denom[col] + acts[col * alphabet_size + idx];
           
            Tp grad = exp(alphas[col] + betas[col] + logpk - logll[mb]);
        
            if (idx == blank_ && t == T-1 && u == U-1) {
                grad -= exp(alphas[col] + logpk - logll[mb]);
            }
            if (idx == blank_ && t < T-1) {
                grad -= exp(alphas[col] + logpk - logll[mb] + betas[col + maxU]);
            }
            if (u < U-1 && idx == labels[u]) {
                grad -= exp(alphas[col] + logpk - logll[mb] + betas[col+1]);
            }
            grads[col * alphabet_size + idx] = grad;

            idx += NT;
        }
    }
}


template<typename ProbT>
class GpuRNNT {
public:
    GpuRNNT(int minibatch, int maxT, int maxU, int alphabet_size, void* workspace, 
            int blank, int num_threads, CUstream stream) :
        minibatch_(minibatch), maxT_(maxT), maxU_(maxU), alphabet_size_(alphabet_size), 
        gpu_workspace(workspace), blank_(blank), num_threads_(num_threads), stream_(stream) {

    };

    GpuRNNT(const GpuRNNT&) = delete;
    GpuRNNT& operator=(const GpuRNNT&) = delete;

    void log_softmax(const ProbT* const acts, ProbT* denom);

    rnntStatus_t compute_cost_and_score(const ProbT*  acts,
                                        ProbT* grad,
                                        ProbT* costs,
                                        const int*  pad_labels,
                                        const int*  label_lengths,
                                        const int*  input_lengths);

    rnntStatus_t cost_and_grad(const ProbT*  acts,
                              ProbT* grad,
                              ProbT* costs,
                              const int*  pad_labels,
                              const int*  label_lengths,
                              const int*  input_lengths);

    rnntStatus_t score_forward(const ProbT*  acts,
                              ProbT* costs,
                              const int*  pad_labels,
                              const int*  label_lengths,
                              const int*  input_lengths);

private:
    int minibatch_;
    int maxT_;
    int maxU_;
    int alphabet_size_; 
    void* gpu_workspace;
    int blank_;
    int num_threads_;
    CUstream stream_;
    
};

template<typename ProbT>
void
GpuRNNT<ProbT>::log_softmax(const ProbT*  acts, ProbT* denom) {

    reduce_max(acts, denom, alphabet_size_, minibatch_ * maxT_ * maxU_, 0, stream_);
    reduce_exp(acts, denom, alphabet_size_, minibatch_ * maxT_ * maxU_, 1, stream_);
}

template<typename ProbT>
rnntStatus_t
GpuRNNT<ProbT>::compute_cost_and_score(const ProbT*  acts,
                                    ProbT* grads,
                                    ProbT* costs,
                                    const int*  labels,
                                    const int*  label_lengths,
                                    const int*  input_lengths) {
    
    
    bool training = (grads != nullptr);
    size_t bytes_used = 0;
    
    ProbT* denom = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
    
    ProbT* alphas = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
    ProbT* betas = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * maxT_ * maxU_ * minibatch_;
    
    ProbT* llForward = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * minibatch_;
    ProbT* llBackward = reinterpret_cast<ProbT*>(static_cast<char*>(gpu_workspace) + bytes_used);
    bytes_used += sizeof(ProbT) * minibatch_;
    
    if (training) {
        cudaMemsetAsync(grads, 0, sizeof(ProbT) * minibatch_ * maxT_ * maxU_ * alphabet_size_, stream_);
    }
    
    log_softmax(acts, denom);

    compute_alphas_kernel<ProbT><<<minibatch_, maxU_, 0, stream_>>>(acts, denom, alphas, llForward, 
        input_lengths, label_lengths, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);
    

    if (training) {
        compute_betas_kernel<ProbT><<<minibatch_, maxU_, 0, stream_>>>(acts, denom, betas, llBackward,
            input_lengths, label_lengths, labels, minibatch_, maxT_, maxU_, alphabet_size_, blank_);

        compute_grad_kernel<128, ProbT><<<minibatch_ * maxT_ * maxU_, 128, 0, stream_>>>(grads, 
            acts, denom, alphas, betas, llForward, input_lengths, label_lengths, labels, 
            minibatch_, maxT_, maxU_, alphabet_size_, blank_);
    }
    
    cudaMemcpyAsync(costs, llForward, sizeof(ProbT) * minibatch_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    
    negp<ProbT><<<1,minibatch_,0,stream_>>>(costs,minibatch_);
   
    return RNNT_STATUS_SUCCESS;
}

template<typename ProbT>
rnntStatus_t
GpuRNNT<ProbT>::cost_and_grad(const ProbT*  acts,
                       ProbT* grads,
                       ProbT* costs,
                       const int*  pad_labels,
                       const int*  label_lengths,
                       const int*  input_lengths) {

    if (acts == nullptr ||
        grads == nullptr || 
        costs == nullptr ||
        pad_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr)
        return RNNT_STATUS_INVALID_VALUE;
    
    return compute_cost_and_score(acts, grads, costs, pad_labels, label_lengths, input_lengths);
}

template<typename ProbT>
rnntStatus_t
GpuRNNT<ProbT>::score_forward(const ProbT*  acts,
                       ProbT* costs,
                       const int*  pad_labels,
                       const int*  label_lengths,
                       const int*  input_lengths) {
    
    if (acts == nullptr ||
        costs == nullptr ||
        pad_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr)
        return RNNT_STATUS_INVALID_VALUE;

    return compute_cost_and_score(acts, nullptr, costs, pad_labels, label_lengths, input_lengths);
}

} // namespace oneflow

#endif // ONEFLOW_USER_KERNELS_RNNT_KERNEL_GPU_H_