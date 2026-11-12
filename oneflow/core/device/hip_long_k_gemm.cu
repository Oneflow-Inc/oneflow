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
#if defined(WITH_CUDA) && defined(USE_HIP)

#include <cublas_v2.h>
#include <string>
#include <vector>
#include "oneflow/core/common/util.h"

namespace {

__host__ __device__ int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Splits n items into the fewest pieces of at most `limit`, all of nearly equal size, so that no
// piece is much narrower than the others.
int64_t BalancedPiece(int64_t n, int64_t limit) { return CeilDiv(n, CeilDiv(n, limit)); }

__device__ float ToFloat(float x) { return x; }
__device__ float ToFloat(__half x) { return __half2float(x); }
__device__ float ToFloat(__hip_bfloat16 x) { return __bfloat162float(x); }
template<typename T>
__device__ T FromFloat(float x);
template<>
__device__ __half FromFloat<__half>(float x) {
  return __float2half(x);
}
template<>
__device__ __hip_bfloat16 FromFloat<__hip_bfloat16>(float x) {
  return __float2bfloat16(x);
}

// The unit roundoff (2^-bits for a significand of that many bits) and smallest normal of each type.
template<typename T>
struct Precision;
template<>
struct Precision<float> {
  static constexpr float kUnitRoundoff = 0x1p-24f;
  static constexpr float kSmallestNormal = 0x1p-126f;
};
template<>
struct Precision<__half> {
  static constexpr float kUnitRoundoff = 0x1p-11f;
  static constexpr float kSmallestNormal = 0x1p-14f;
};
template<>
struct Precision<__hip_bfloat16> {
  static constexpr float kUnitRoundoff = 0x1p-8f;
  static constexpr float kSmallestNormal = 0x1p-126f;
};

// At most how far x, a normal value rounded to P, can be from the value it was rounded from: half
// an ulp of P, from x's exponent. 0 below P's smallest normal; inf for inf.
template<typename P>
__device__ float HalfUlp(float x) {
  if (!(fabsf(x) >= Precision<P>::kSmallestNormal)) { return 0.0f; }
  return __int_as_float(__float_as_int(x) & 0x7f800000) * Precision<P>::kUnitRoundoff;
}

constexpr int kRun = 8;

// kRun consecutive k indices of op(A) or op(B) as floats, from p and every step-th element after
// it: one 16-byte load where they are contiguous and aligned.
template<typename T>
__device__ void LoadRun(const T* p, int64_t step, float* x) {
  static_assert(kRun * sizeof(T) == sizeof(uint4), "a run is one 16-byte load");
  if (step == 1 && reinterpret_cast<uintptr_t>(p) % sizeof(uint4) == 0) {
    const uint4 word = *reinterpret_cast<const uint4*>(p);
    const T* values = reinterpret_cast<const T*>(&word);
    for (int q = 0; q < kRun; ++q) { x[q] = ToFloat(values[q]); }
  } else {
    for (int q = 0; q < kRun; ++q) { x[q] = ToFloat(p[q * step]); }
  }
}

constexpr int kStoreRows = 32;
constexpr int kStoreCols = 8;
constexpr int kStoreSteps = 32;
// When a tile's output blocks hold fewer than 16 elements, which on random data are computed
// again often, and there are at most kSplitBlocks / 2 of them, StoreTile runs up to kMaxSplits
// thread blocks per output block, so that computing its elements again over all of k is spread
// over the GPU rather than left to one compute unit. Blocks of 16 elements or more seldom are, and
// do not pay for the split.
constexpr int64_t kSplitBlocks = 512;
constexpr int64_t kMaxSplits = 64;

int64_t StoreSplits(int64_t rows, int64_t cols, int64_t count) {
  if (std::min<int64_t>(rows, kStoreRows) * std::min<int64_t>(cols, kStoreCols) >= 16) { return 1; }
  const int64_t output_blocks = CeilDiv(rows, kStoreRows) * CeilDiv(cols, kStoreCols) * count;
  return std::max<int64_t>(1, std::min(kMaxSplits, kSplitBlocks / output_blocks));
}

// d = (the sum of the slices of tile) + beta * d for `count` rows x cols blocks of column-major
// matrices, block b at d + b * stride_d. tile holds `slices` x count column-major rows x cols
// matrices back to back, slice s of block b at index s * count + b. The sum is in float and d is
// rounded once; d is not read when beta is 0, as in BLAS. Run with kStoreRows x kStoreCols threads.
//
// A 16-bit slice limits each k-chunk's product to the range of the type, and rounds it relative to
// its own size: a float16 chunk sum above 65504 is inf even when the other chunks cancel it, and
// chunk sums far larger than the results they cancel to lose those results' precision. So unless a
// is null, the elements of each kStoreRows x kStoreCols block are computed again from a and b
// (blocks at stride_a and stride_b) as float dot products over all of k, as a single GEMM
// accumulating in float gives them, when any element's slices do not sum to a finite value or
// could be off by more than 3 units of roundoff of T times the block's largest result. That bound
// is the sum of half an ulp of each slice and of each partial sum in float; all it leaves out is
// half the smallest subnormal step of P for each slice below P's smallest normal. Inputs that hold
// inf or NaN give non-finite results either way. With splits > 1, each output block has that many
// thread blocks, which each take a share of k and leave their sums in `partials`; the last of them
// to finish, counted in `arrived` (zero on entry), adds them in order and writes d.
template<typename T, typename P>
__global__ void __launch_bounds__(kStoreRows* kStoreCols)
    StoreTile(int64_t rows, int64_t cols, int64_t count, int64_t slices, const P* tile, float beta,
              T* d, int64_t ldd, int64_t stride_d, bool trans_a, bool trans_b, int64_t k,
              float alpha, const T* a, int64_t lda, int64_t stride_a, const T* b, int64_t ldb,
              int64_t stride_b, int64_t splits, double* partials, unsigned int* arrived) {
  __shared__ float shared_a[kStoreRows][kStoreSteps + 1];
  __shared__ float shared_b[kStoreSteps][kStoreCols + 1];
  __shared__ float wave_max[kStoreRows * kStoreCols];
  __shared__ double partial[kStoreRows * kStoreCols];
  __shared__ bool last;
  const int64_t size = rows * cols;
  const int64_t blocks_m = CeilDiv(rows, kStoreRows);
  const int64_t blocks = blocks_m * CeilDiv(cols, kStoreCols);
  const int tid = threadIdx.y * kStoreRows + threadIdx.x;
  for (int64_t u = blockIdx.x; u < blocks * count * splits; u += gridDim.x) {
    const int64_t t = splits == 1 ? u : u / splits;
    const int64_t split = splits == 1 ? 0 : u % splits;
    const int64_t c = t / blocks;
    const int64_t i0 = t % blocks % blocks_m * kStoreRows;
    const int64_t j0 = t % blocks / blocks_m * kStoreCols;
    const int64_t r = i0 + threadIdx.x;
    const int64_t col = j0 + threadIdx.y;
    const bool inside = r < rows && col < cols;
    float v = 0.0f;
    float error = 0.0f;
    if (inside) {
      for (int64_t s = 0; s < slices; ++s) {
        const float x = ToFloat(tile[c * size + r + col * rows + s * size * count]);
        v += x;
        // The slice's rounding to P, and the addition's rounding to float.
        error += HalfUlp<P>(x) + HalfUlp<float>(v);
      }
    }
    bool again = false;
    if (a != nullptr) {
      float largest = inside && isfinite(v) ? fabsf(v) : 0.0f;
      for (int lane = warpSize / 2; lane > 0; lane /= 2) {
        largest = fmaxf(largest, __shfl_xor(largest, lane));
      }
      if (tid % warpSize == 0) { wave_max[tid / warpSize] = largest; }
      __syncthreads();
      for (int w = 0; w < kStoreRows * kStoreCols / warpSize; ++w) {
        largest = fmaxf(largest, wave_max[w]);
      }
      const float budget = 3.0f * Precision<T>::kUnitRoundoff * largest;
      again = __syncthreads_or(inside && !(isfinite(v) && error <= budget));
    }
    if (!again && split > 0) { continue; }
    if (again) {
      const T* block_a = a + c * stride_a;
      const T* block_b = b + c * stride_b;
      const int64_t share = CeilDiv(CeilDiv(k, splits), kStoreSteps) * kStoreSteps;
      const int64_t begin = std::min(k, split * share);
      const int64_t end = std::min(k, begin + share);
      const int valid_rows = static_cast<int>(std::min<int64_t>(rows - i0, kStoreRows));
      const int valid_cols = static_cast<int>(std::min<int64_t>(cols - j0, kStoreCols));
      double sum = 0.0;
      if (2 * valid_rows * valid_cols > kStoreRows * kStoreCols) {
        for (int64_t l0 = begin; l0 < end; l0 += kStoreSteps) {
          // Each step reads op(A)(i0.., l0..) and op(B)(l0.., j0..) along their contiguous index.
          for (int q = 0; q < kStoreRows * kStoreSteps / (kStoreRows * kStoreCols); ++q) {
            const int e = q * kStoreRows * kStoreCols + tid;
            const int ar = trans_a ? e / kStoreSteps : e % kStoreRows;
            const int al = trans_a ? e % kStoreSteps : e / kStoreRows;
            const bool ok = i0 + ar < rows && l0 + al < end;
            shared_a[ar][al] = ok ? ToFloat(trans_a ? block_a[l0 + al + (i0 + ar) * lda]
                                                    : block_a[i0 + ar + (l0 + al) * lda])
                                  : 0.0f;
          }
          const int bl = trans_b ? tid / kStoreCols : tid % kStoreSteps;
          const int bc = trans_b ? tid % kStoreCols : tid / kStoreSteps;
          const bool ok = j0 + bc < cols && l0 + bl < end;
          shared_b[bl][bc] = ok ? ToFloat(trans_b ? block_b[j0 + bc + (l0 + bl) * ldb]
                                                  : block_b[l0 + bl + (j0 + bc) * ldb])
                                : 0.0f;
          __syncthreads();
          float part = 0.0f;
          for (int q = 0; q < kStoreSteps; ++q) {
            part = fmaf(shared_a[threadIdx.x][q], shared_b[q][threadIdx.y], part);
          }
          sum += part;
          __syncthreads();
        }
      } else {
        // At most half the threads have an element: give each element `parts` threads, the
        // power-of-two share of the block, which take every parts-th k index and are added after.
        int block_rows = 1;
        while (block_rows < valid_rows) { block_rows *= 2; }
        int block_cols = 1;
        while (block_cols < valid_cols) { block_cols *= 2; }
        const int parts = kStoreRows * kStoreCols / (block_rows * block_cols);
        const int part = tid % parts;
        const int er = tid / parts % block_rows;
        const int ec = tid / parts / block_rows;
        if (er < valid_rows && ec < valid_cols) {
          const T* row_a = block_a + (i0 + er) * (trans_a ? lda : 1);
          const int64_t step_a = trans_a ? 1 : lda;
          const T* col_b = block_b + (j0 + ec) * (trans_b ? 1 : ldb);
          const int64_t step_b = trans_b ? ldb : 1;
          // Runs of kRun consecutive k indices, every parts-th run to this thread.
          const int64_t runs = (end - begin) / kRun;
          float part_sum = 0.0f;
          int terms = 0;
          for (int64_t j = part; j < runs; j += parts) {
            float run_a[kRun];
            float run_b[kRun];
            const int64_t l = begin + j * kRun;
            LoadRun(row_a + l * step_a, step_a, run_a);
            LoadRun(col_b + l * step_b, step_b, run_b);
            for (int q = 0; q < kRun; ++q) { part_sum = fmaf(run_a[q], run_b[q], part_sum); }
            terms += kRun;
            if (terms == kStoreSteps) {
              sum += part_sum;
              part_sum = 0.0f;
              terms = 0;
            }
          }
          if (part == 0) {
            for (int64_t l = begin + runs * kRun; l < end; ++l) {
              part_sum = fmaf(ToFloat(row_a[l * step_a]), ToFloat(col_b[l * step_b]), part_sum);
            }
          }
          sum += part_sum;
        }
        partial[tid] = sum;
        __syncthreads();
        for (int width = parts / 2; width > 0; width /= 2) {
          if (part < width) { partial[tid] += partial[tid + width]; }
          __syncthreads();
        }
        if (inside) { sum = partial[(threadIdx.x + threadIdx.y * block_rows) * parts]; }
      }
      if (splits > 1) {
        double* sums = partials + t * splits * (kStoreRows * kStoreCols);
        sums[split * (kStoreRows * kStoreCols) + tid] = sum;
        __threadfence();
        __syncthreads();
        if (tid == 0) { last = atomicAdd(arrived + t, 1u) == splits - 1; }
        __syncthreads();
        if (!last) { continue; }
        __threadfence();
        sum = 0.0;
        for (int64_t s = 0; s < splits; ++s) {
          sum += __hip_atomic_load(sums + s * (kStoreRows * kStoreCols) + tid, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
        }
      }
      v = alpha * static_cast<float>(sum);
    }
    if (inside) {
      T* p = d + c * stride_d + r + col * ldd;
      if (beta != 0.0f) { v += beta * ToFloat(*p); }
      *p = FromFloat<T>(v);
    }
  }
}

// The k-chunk products of one tile, each written to its own slice of `tile` in the type of D: one
// call per chunk with the batch folded in, or one call per batch with the chunks folded in (plus
// one for a shorter last chunk), whichever needs fewer calls.
template<typename T>
hipblasStatus_t ChunkProducts(hipblasHandle_t handle, hipblasLtHandle_t lt_handle,
                              hipStream_t stream, void* workspace, size_t workspace_size,
                              hipblasOperation_t trans_a, hipblasOperation_t trans_b, int64_t rows,
                              int64_t cols, int64_t k, int64_t chunk, float alpha, const T* a,
                              int64_t lda, int64_t stride_a, const T* b, int64_t ldb,
                              int64_t stride_b, int64_t count, T* tile, hipDataType type) {
  const int64_t size = rows * cols;
  const int64_t full_chunks = k / chunk;
  const int64_t last_chunk = k % chunk;
  const int64_t slices = full_chunks + (last_chunk > 0 ? 1 : 0);
  // Element offsets of k index 1 in A and B.
  const int64_t k_step_a = trans_a == HIPBLAS_OP_N ? lda : 1;
  const int64_t k_step_b = trans_b == HIPBLAS_OP_N ? 1 : ldb;
  // rocBLAS when it would not split the chunk into shorter k-chunks of its own, which it rounds to
  // D one by one (see OfUseLongKGemm); it is the library ordinary GEMMs use, and on gfx1100 it is
  // faster than hipBLASLt for large outputs. hipBLASLt otherwise.
  const bool rocblas = std::max(lda, ldb) * chunk < (int64_t{1} << 30);
  const float zero = 0.0f;
  const auto Gemm = [&](int64_t kk, const T* ga, int64_t sa, const T* gb, int64_t sb, T* gd,
                        int64_t sd, int64_t batch) {
    if (rocblas) {
      return hipblasGemmStridedBatchedEx(handle, trans_a, trans_b, rows, cols, kk, &alpha, ga, type,
                                         lda, sa, gb, type, ldb, sb, &zero, gd, type, rows, sd,
                                         batch, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    }
    return OfHipblasLtGemm(lt_handle, stream, workspace, workspace_size, trans_a, trans_b, rows,
                           cols, kk, alpha, ga, lda, sa, gb, ldb, sb, 0.0f, gd, type, type, rows,
                           sd, batch);
  };
  hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
  if (slices <= count * (last_chunk > 0 ? 2 : 1)) {
    for (int64_t s = 0; s < slices && status == HIPBLAS_STATUS_SUCCESS; ++s) {
      status = Gemm(std::min(chunk, k - s * chunk), a + s * chunk * k_step_a, stride_a,
                    b + s * chunk * k_step_b, stride_b, tile + s * count * size, size, count);
    }
  } else {
    for (int64_t i = 0; i < count && status == HIPBLAS_STATUS_SUCCESS; ++i) {
      if (full_chunks > 0) {
        status = Gemm(chunk, a + i * stride_a, chunk * k_step_a, b + i * stride_b, chunk * k_step_b,
                      tile + i * size, count * size, full_chunks);
      }
      if (status == HIPBLAS_STATUS_SUCCESS && last_chunk > 0) {
        status = Gemm(last_chunk, a + i * stride_a + full_chunks * chunk * k_step_a, 0,
                      b + i * stride_b + full_chunks * chunk * k_step_b, 0,
                      tile + (full_chunks * count + i) * size, 0, 1);
      }
    }
  }
  return status;
}

// D in tiles of rows x cols x count batches. When chunked, k is cut into chunks of `chunk` (the
// last may be shorter) whose products are rounded to the type of D in their own slices of the tile,
// and StoreTile sums the slices in float (and computes again any element they leave non-finite).
// Where hipBLASLt has no algorithm for a tile, and throughout when not chunked, rocBLAS gemm_ex
// writes the tile's whole product with a float D, which it accumulates in float over its own
// k-chunks; the tile holds either.
template<typename T>
hipblasStatus_t TiledGemm(hipblasHandle_t handle, hipblasLtHandle_t lt_handle, hipStream_t stream,
                          void* workspace, size_t workspace_size, void* tile, void* split_space,
                          int64_t tile_m, int64_t tile_n, int64_t tile_count, bool chunked,
                          int64_t chunk, hipblasOperation_t trans_a, hipblasOperation_t trans_b,
                          int64_t m, int64_t n, int64_t k, float alpha, const T* a, int64_t lda,
                          int64_t stride_a, const T* b, int64_t ldb, int64_t stride_b, float beta,
                          T* d, hipDataType type, int64_t ldd, int64_t stride_d,
                          int64_t batch_count) {
  const int64_t slices = CeilDiv(k, chunk);
  const float zero = 0.0f;
  for (int64_t batch = 0; batch < batch_count; batch += tile_count) {
    for (int64_t j = 0; j < n; j += tile_n) {
      for (int64_t i = 0; i < m; i += tile_m) {
        const int64_t rows = std::min(tile_m, m - i);
        const int64_t cols = std::min(tile_n, n - j);
        const int64_t count = std::min(tile_count, batch_count - batch);
        const T* tile_a = a + batch * stride_a + (trans_a == HIPBLAS_OP_N ? i : i * lda);
        const T* tile_b = b + batch * stride_b + (trans_b == HIPBLAS_OP_N ? j * ldb : j);
        T* tile_d = d + batch * stride_d + i + j * ldd;
        const dim3 threads(kStoreRows, kStoreCols);
        // One block per output block and split: StoreTile synchronizes its threads for each.
        const int64_t output_blocks = CeilDiv(rows, kStoreRows) * CeilDiv(cols, kStoreCols) * count;
        const int64_t splits = chunked ? StoreSplits(rows, cols, count) : 1;
        const int64_t blocks = std::min<int64_t>(output_blocks * splits, int64_t{1} << 20);
        double* partials = static_cast<double*>(split_space);
        unsigned int* arrived =
            reinterpret_cast<unsigned int*>(partials + kSplitBlocks * kStoreRows * kStoreCols);
        const bool ta = trans_a != HIPBLAS_OP_N;
        const bool tb = trans_b != HIPBLAS_OP_N;
        hipblasStatus_t status = HIPBLAS_STATUS_NOT_SUPPORTED;
        if (chunked) {
          status = ChunkProducts(handle, lt_handle, stream, workspace, workspace_size, trans_a,
                                 trans_b, rows, cols, k, chunk, alpha, tile_a, lda, stride_a,
                                 tile_b, ldb, stride_b, count, static_cast<T*>(tile), type);
          if (status == HIPBLAS_STATUS_SUCCESS && splits > 1
              && hipMemsetAsync(arrived, 0, output_blocks * sizeof(unsigned int), stream)
                     != hipSuccess) {
            return HIPBLAS_STATUS_EXECUTION_FAILED;
          }
          if (status == HIPBLAS_STATUS_SUCCESS) {
            StoreTile<T, T><<<blocks, threads, 0, stream>>>(
                rows, cols, count, slices, static_cast<const T*>(tile), beta, tile_d, ldd, stride_d,
                ta, tb, k, alpha, tile_a, lda, stride_a, tile_b, ldb, stride_b, splits, partials,
                arrived);
          }
        }
        if (status == HIPBLAS_STATUS_NOT_SUPPORTED) {
          float* float_tile = static_cast<float*>(tile);
          status = HIPBLAS_STATUS_SUCCESS;
          for (int64_t c = 0; c < count && status == HIPBLAS_STATUS_SUCCESS; ++c) {
            status = hipblasGemmEx(handle, trans_a, trans_b, rows, cols, k, &alpha,
                                   tile_a + c * stride_a, type, lda, tile_b + c * stride_b, type,
                                   ldb, &zero, float_tile + c * rows * cols, HIP_R_32F, rows,
                                   HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
          }
          if (status == HIPBLAS_STATUS_SUCCESS) {
            StoreTile<T, float><<<blocks, threads, 0, stream>>>(
                rows, cols, count, 1, float_tile, beta, tile_d, ldd, stride_d, ta, tb, k, alpha,
                static_cast<const T*>(nullptr), lda, stride_a, tile_b, ldb, stride_b, 1, nullptr,
                nullptr);
          }
        }
        if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
        if (hipGetLastError() != hipSuccess) { return HIPBLAS_STATUS_EXECUTION_FAILED; }
      }
    }
  }
  return HIPBLAS_STATUS_SUCCESS;
}

}  // namespace

// The gfx11 and gfx12 matrix instructions (WMMA) do not round the float accumulator to nearest: a
// chain of 2^20 float16 products drifts about 100 (gfx1100) or 50 (gfx1201) times further than a
// chain of fp32 FMAs, while gfx9 (MFMA) keeps fp32 accuracy.
bool OfMatrixCoreAccumulationDrifts() {
  static const std::vector<bool> drifts = [] {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) { count = 0; }
    std::vector<bool> result(count);
    for (int i = 0; i < count; ++i) {
      hipDeviceProp_t prop;
      if (hipGetDeviceProperties(&prop, i) != hipSuccess) { continue; }
      const std::string arch = prop.gcnArchName;
      result[i] = arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0;
    }
    return result;
  }();
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess || device >= static_cast<int>(drifts.size())) {
    return false;
  }
  return drifts[device];
}

hipblasStatus_t OfLongKGemm(hipblasHandle_t handle, hipblasLtHandle_t lt_handle, hipStream_t stream,
                            void* workspace, size_t workspace_size, hipblasOperation_t trans_a,
                            hipblasOperation_t trans_b, int64_t m, int64_t n, int64_t k,
                            float alpha, const void* a, int64_t lda, int64_t stride_a,
                            const void* b, int64_t ldb, int64_t stride_b, float beta, void* d,
                            hipDataType type, int64_t ldd, int64_t stride_d, int64_t batch_count) {
  // BLAS quick return: an empty D needs no write.
  if (m == 0 || n == 0 || batch_count == 0) { return HIPBLAS_STATUS_SUCCESS; }
  const bool use_hipblaslt =
      oneflow::ParseBooleanFromEnv("ONEFLOW_EP_CUDA_LONG_K_GEMM_USE_HIPBLASLT", true);
  // Where the accumulator drifts, no library gets a GEMM with k above kOfLongKGemmChain. That also
  // keeps clear of kernels that are wrong only at large k: on gfx1201 the heuristic's first choice
  // for k-major A and small outputs is off by tens of float16 ulp for some k above 2^17, and a
  // hipBLASLt split-k does not change that.
  const bool chunked = use_hipblaslt && OfMatrixCoreAccumulationDrifts();
  if (use_hipblaslt && !chunked) {
    const hipblasStatus_t status = OfHipblasLtGemm(
        lt_handle, stream, workspace, workspace_size, trans_a, trans_b, m, n, k, alpha, a, lda,
        stride_a, b, ldb, stride_b, beta, d, type, type, ldd, stride_d, batch_count);
    if (status != HIPBLAS_STATUS_NOT_SUPPORTED) { return status; }
  }
  if (type != HIP_R_16F && type != HIP_R_16BF) { return HIPBLAS_STATUS_NOT_SUPPORTED; }
  // Equal chunks of about 2^13, which measured more accurate than longer ones and no slower; but
  // at most 16 of them unless that would make a chunk longer than kOfLongKGemmChain, so that the
  // slices need little memory.
  const int64_t slices =
      chunked ? std::max(CeilDiv(k, kOfLongKGemmChain), std::min<int64_t>(CeilDiv(k, 1 << 13), 16))
              : 1;
  const int64_t chunk = CeilDiv(k, slices);
  // Tiles of at most ONEFLOW_EP_CUDA_LONG_K_GEMM_MAX_TILE_SIZE elements of the type of D in all,
  // which hold each output element once per k-chunk (or once in float for the rocBLAS path); in
  // the workspace when that is large enough and a temporary allocation otherwise. hipBLASLt gets
  // the rest of the workspace.
  const int64_t max_tile_size = std::max<int64_t>(
      1,
      oneflow::ParseIntegerFromEnv("ONEFLOW_EP_CUDA_LONG_K_GEMM_MAX_TILE_SIZE", int64_t{1} << 25));
  const int64_t units = std::max<int64_t>(slices, sizeof(float) / 2);
  const int64_t max_outputs = std::max<int64_t>(1, max_tile_size / units);
  int64_t tile_m = m;
  int64_t tile_n = n;
  int64_t tile_count = 1;
  if (m * n <= max_outputs) {
    tile_count = BalancedPiece(batch_count, max_outputs / (m * n));
  } else {
    tile_m = BalancedPiece(m, max_outputs);
    tile_n = BalancedPiece(n, std::max<int64_t>(1, max_outputs / tile_m));
  }
  // Space for StoreTile's split sums when some tile is split; if any is, the last tile in each
  // dimension, the smallest, is.
  const int64_t last_m = m - (CeilDiv(m, tile_m) - 1) * tile_m;
  const int64_t last_n = n - (CeilDiv(n, tile_n) - 1) * tile_n;
  const int64_t last_count = batch_count - (CeilDiv(batch_count, tile_count) - 1) * tile_count;
  const bool split = chunked && StoreSplits(last_m, last_n, last_count) > 1;
  const size_t split_bytes = split ? kSplitBlocks * kStoreRows * kStoreCols * sizeof(double)
                                         + kSplitBlocks * sizeof(unsigned int)
                                   : 0;
  // 256-byte aligned, as hipBLASLt expects of its workspace.
  const size_t slice_bytes = (tile_m * tile_n * tile_count * units * 2 + 255) / 256 * 256;
  const size_t tile_bytes = slice_bytes + (split_bytes + 255) / 256 * 256;
  void* tile = workspace;
  void* lt_workspace = workspace;
  size_t lt_workspace_size = workspace_size;
  if (workspace_size >= tile_bytes) {
    lt_workspace = static_cast<char*>(workspace) + tile_bytes;
    lt_workspace_size = workspace_size - tile_bytes;
  } else if (hipMallocAsync(&tile, tile_bytes, stream) != hipSuccess) {
    return HIPBLAS_STATUS_ALLOC_FAILED;
  }
  hipblasStatus_t status;
  if (type == HIP_R_16F) {
    status = TiledGemm(handle, lt_handle, stream, lt_workspace, lt_workspace_size, tile,
                       static_cast<char*>(tile) + slice_bytes, tile_m, tile_n, tile_count, chunked,
                       chunk, trans_a, trans_b, m, n, k, alpha, static_cast<const __half*>(a), lda,
                       stride_a, static_cast<const __half*>(b), ldb, stride_b, beta,
                       static_cast<__half*>(d), type, ldd, stride_d, batch_count);
  } else {
    status =
        TiledGemm(handle, lt_handle, stream, lt_workspace, lt_workspace_size, tile,
                  static_cast<char*>(tile) + slice_bytes, tile_m, tile_n, tile_count, chunked,
                  chunk, trans_a, trans_b, m, n, k, alpha, static_cast<const __hip_bfloat16*>(a),
                  lda, stride_a, static_cast<const __hip_bfloat16*>(b), ldb, stride_b, beta,
                  static_cast<__hip_bfloat16*>(d), type, ldd, stride_d, batch_count);
  }
  if (tile != workspace) { (void)hipFreeAsync(tile, stream); }
  return status;
}

#endif  // defined(WITH_CUDA) && defined(USE_HIP)
