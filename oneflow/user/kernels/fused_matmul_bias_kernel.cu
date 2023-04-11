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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/cublas_fused_mlp_util.cuh"

// same with cublas_fused_mlp_util.cuh
#if CUDA_VERSION >= 11020

namespace oneflow {

namespace {

#define PLUGIN_CUBLASASSERT(__call) OF_CUBLAS_CHECK(__call)
#define PLUGIN_CUASSERT(__call) OF_CUDA_CHECK(__call)

auto constexpr kNB_ALGO_COMBINATIONS = 6000;
auto constexpr kNB_ALGO_IDS = 40;
auto constexpr kPRINT_ALGOS = 8;
auto constexpr kNB_KERNEL_REPEATS = 10;
auto constexpr kTHREADS_PER_BLOCK = 1024;

char const* const matmulTileName[] = {
    "UNDEF",
    "8x8",
    "8x16",
    "16x8",
    "8x32",
    "16x16",
    "32x8",
    "8x64",
    "16x32",
    "32x16",
    "64x8",
    "32x32",
    "32x64",
    "64x32",
    "32x128",
    "64x64",
    "128x32",
    "64x128",
    "128x64",
    "64x256",
    "128x128",
    "256x64",
    "64x512",
    "128x256",
    "256x128",
    "512x64",
};

typedef struct customMatMultPerfType_t
{
    static constexpr float kMAX_TIME = 1000000.F;
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t status;
    float time{kMAX_TIME};
    size_t workspaceSize; // actual memory workspace needed
    cublasMath_t mathMode;
    cublasLtReductionScheme_t reductionScheme;
    int32_t customOption;
    float wavesCount;
} customMatmulPerf_t;

namespace
{
char const* const kFC_VERSION{"1"};
char const* const kFC_NAME{"CustomFCPluginDynamic"};
constexpr size_t kMAX_WORKSPACE_BYTES = 4 * 1024 * 1024; // 4MiB
} // namespace

struct AlgoProps
{
    int32_t algoId;
    int32_t tile;
    int32_t swizzle;
    int32_t customOption;
    int32_t numSplitsK;
    int32_t reductionScheme;
    uint64_t numericImpl;

    void populate(cublasLtMatmulAlgo_t const& algo)
    {
        cublasLtMatmulAlgo_t const* matmulAlgo = &algo;
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), nullptr));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            matmulAlgo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &numericImpl, sizeof(numericImpl), nullptr));
    }
};

// Utility function to print customMatmulPerf_t structure
static void printPerfStructure(customMatmulPerf_t const& perf, int32_t const m, int32_t const n, int32_t const k)
{
    AlgoProps p;
    p.populate(perf.algo);
    // Calculate GFLOPS
    double timeAvg
        = perf.time * 1e-3; // Convert to seconds. It has been divided by kNB_KERNEL_REPEATS in customMatmulRun().
    double gflop = (2 * static_cast<uint64_t>(m * n) * k) * 1e-9; // Real

    std::cout << "Algo=" << p.algoId << " Tile=" << p.tile << " (" << matmulTileName[p.tile] << ") K=" << p.numSplitsK
                << " Red.Sch.=" << p.reductionScheme << " Swiz=" << p.swizzle << " Cust=" << p.customOption
                << " Stat=" << perf.status << " Time=" << perf.time << " WSbytes=" << perf.workspaceSize
                << " math=" << p.numericImpl << " waves=" << perf.wavesCount << "GFlops=" << (gflop / timeAvg)
                << std::endl;
}

static bool timeCompare(customMatmulPerf_t const& perf_a, customMatmulPerf_t const& perf_b)
{
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}

static cublasStatus_t customMatmulRun(cublasLtHandle_t ltHandle, // to get the capabilities (required a GPU)
    cublasLtMatmulDesc_t operationDesc, void const* alpha,       // host or device pointer
    void const* A, cublasLtMatrixLayout_t Adesc, void const* B, cublasLtMatrixLayout_t Bdesc,
    void const* beta, // host or device pointer
    void const* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulAlgo_t const& algo, void* workSpace, size_t workSpaceSizeInBytes, customMatmulPerf_t& perfResults,
    cudaStream_t stream, cudaEvent_t& startEvent, cudaEvent_t& stopEvent)
{
    cublasLtMatmulHeuristicResult_t heurResult;

    // Looping over the Algo
    cublasStatus_t algoStatus
        = cublasLtMatmulAlgoCheck(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);

    if (algoStatus == CUBLAS_STATUS_SUCCESS)
    {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes)
        {
            if (cudaEventRecord(startEvent, stream) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            for (int32_t loop = 0; loop < kNB_KERNEL_REPEATS; loop++)
            {
                cublasStatus_t oneRunStatus = cublasLtMatmul(ltHandle, operationDesc, alpha, // host or device pointer
                    A, Adesc, B, Bdesc, beta,                                                // host or device pointer
                    C, Cdesc, D, Ddesc, &algo, workSpace, workSpaceSizeInBytes, stream);
                if (oneRunStatus != CUBLAS_STATUS_SUCCESS)
                {
                    algoStatus = oneRunStatus;
                    break;
                }
            }
            if (cudaEventRecord(stopEvent, stream) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            if (cudaEventSynchronize(stopEvent) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            float time;
            if (cudaEventElapsedTime(&time, startEvent, stopEvent) != cudaSuccess)
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            // For the moment only add successful findings
            perfResults.algo = algo;
            perfResults.time = time / kNB_KERNEL_REPEATS; // Average time
            perfResults.workspaceSize = heurResult.workspaceSize;
            perfResults.wavesCount = heurResult.wavesCount;
        }
        else
        {
            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED; // Not enough workspace
        }
    }
    return algoStatus;
}

// Sample wrapper running through multiple algo and config attributes
// combination for single precision gemm using cublasLt low-level API
void LtGemmSearch(cublasLtHandle_t ltHandle, cublasOperation_t transa, cublasOperation_t transb, int32_t const& m,
    int32_t const& n, int32_t const& k, void const* alpha,                                  // host pointer
    void const* A, int32_t const& lda, void const* B, int32_t const& ldb, void const* beta, // host pointer
    void* C, int32_t const& ldc, void* workSpace, size_t workSpaceSize,
#if CUBLAS_VER_MAJOR < 11
    cudaDataType_t computeType,
#else
    cublasComputeType_t computeType,
#endif
    cudaDataType_t scaleType, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype,
    std::vector<customMatmulPerf_t>& perfResults)
{

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr;
    cublasLtMatrixLayout_t Bdesc = nullptr;
    cublasLtMatrixLayout_t Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    cudaStream_t stream = nullptr;

    // SplitK value that we are going to try when SplitK is supported for a given algo.
    int32_t const splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};

    // Let try a fixed number of combinations
    int32_t algoCount = 0;
    int32_t nbAlgoIds = 0;
    int32_t algoIdA[kNB_ALGO_IDS];

    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceCreate(&preference));
    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workSpaceSize, sizeof(workSpaceSize)));

    uint64_t const numericImplPrefer
        = Ctype == CUDA_R_16F ? CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA : CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA;
    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_IMPL_MASK, &numericImplPrefer, sizeof(numericImplPrefer)));

    // Create operation descriptor; see cublasLtMatmulDescAttributes_t for details
    // about defaults; here we just need to set the transforms for A and B
#if CUBLAS_VER_MAJOR < 11
    PLUGIN_CUBLASASSERT(cublasLtMatmulDescCreate(&operationDesc, computeType));
#else
    PLUGIN_CUBLASASSERT(cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));
#endif
    PLUGIN_CUBLASASSERT(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    PLUGIN_CUBLASASSERT(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

    // Create matrix descriptors. We are good with the details here so no need to
    // set any extra attributes
    PLUGIN_CUBLASASSERT(
        cublasLtMatrixLayoutCreate(&Adesc, Atype, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    PLUGIN_CUBLASASSERT(
        cublasLtMatrixLayoutCreate(&Bdesc, Btype, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc));

    // Request the 4 first AlgoId available for SGEMM ( computeType = scaleType =
    // Atype = Btype = Ctype = Dtype = CUDA_R_32F)
    PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoGetIds(
        ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, kNB_ALGO_IDS, algoIdA, &nbAlgoIds));

    std::cout << "Number of algos" << nbAlgoIds << std::endl;

    // Create CUDA event to time the execution time of each algo
    PLUGIN_CUASSERT(cudaEventCreate(&startEvent, cudaEventBlockingSync));
    PLUGIN_CUASSERT(cudaEventCreate(&stopEvent, cudaEventBlockingSync));

    // Loop over the Algo IDs
    for (int32_t idx = 0; (idx < nbAlgoIds) && (algoCount < kNB_ALGO_COMBINATIONS); idx++)
    {
        cublasLtMatmulAlgo_t algo;
        size_t sizeWritten = 0;
        // Initialize algo structure with given Algp ID.
        status
            = cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            continue;
        }

        uint64_t numericImpl = -1;
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &numericImpl, sizeof(numericImpl), nullptr));
        if (Ctype == CUDA_R_32F && numericImpl == CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA)
        {
            // skip HMMA-fp32accu kernels
            continue;
        }

        // Query the tiles enums supported by that algo
        PLUGIN_CUBLASASSERT(
            cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sizeWritten));
        int32_t nbTiles = int32_t(sizeWritten / sizeof(int32_t));
        int32_t* tileA = new int32_t[nbTiles == 0 ? 1 : nbTiles];
        if (nbTiles == 0)
        {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles = 1;
        }

        int32_t splitkSupport;
        int32_t redMask;
        int32_t swizzlingMax;
        int32_t customOptionMax;
        int32_t epilogueMask;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the
        // different combinations
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int32_t) * nbTiles, &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten));
        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten));

        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_EPILOGUE_MASK, &epilogueMask, sizeof(epilogueMask), &sizeWritten));

        // Loop over the different tiles
        for (int32_t tileIdx = 0; tileIdx < nbTiles; tileIdx++)
        {
            // Loop over the different custom option if any
            for (int32_t customOption = 0; customOption <= customOptionMax; customOption++)
            {
                PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption)));
                // Loop over the CTAs swizzling support
                for (int32_t k = 0; k <= swizzlingMax; k++)
                {
                    int32_t splitkTrial = 0;
                    if (splitkSupport)
                    {
                        splitkTrial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                    }
                    // Loop over the splitK value over a fixed sequence splitKSequenceA in
                    // addition to the case where splitK is not enabled
                    for (int32_t l = 0; (l < (1 + splitkTrial)) && (algoCount < kNB_ALGO_COMBINATIONS); l++)
                    {
                        // Setup attribute of the algo to run
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx])));
                        int32_t splitK_val = 0;
                        int32_t redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val)));
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)));
                        PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int32_t)));

                        if (l > 0)
                        { // Split-K case
                            splitK_val = splitKSequenceA[l - 1];
                            PLUGIN_CUBLASASSERT(
                                cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                    &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1])));
                            // Going over all the reduction scheme
                            for (redScheme = 1; redScheme < static_cast<int32_t>(CUBLASLT_REDUCTION_SCHEME_MASK)
                                 && (algoCount < kNB_ALGO_COMBINATIONS);
                                 redScheme = redScheme << 1)
                            {
                                if (redScheme & redMask)
                                {
                                    PLUGIN_CUBLASASSERT(cublasLtMatmulAlgoConfigSetAttribute(
                                        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(redScheme)));

                                    status = customMatmulRun(ltHandle, operationDesc, alpha, // host or device pointer
                                        A, Adesc, B, Bdesc, beta,                            // host or device pointer
                                        C, Cdesc, C, Cdesc, algo, workSpace, workSpaceSize, perfResults[algoCount],
                                        stream, startEvent, stopEvent);
                                    perfResults[algoCount].status = status;
                                    if (status == CUBLAS_STATUS_SUCCESS)
                                    {
                                        algoCount++;
                                    }
                                } // end if
                            }     // end for
                        }
                        else
                        { // Non-splitK case
                            // if user preference is ok with workspace
                            if (algoCount < kNB_ALGO_COMBINATIONS)
                            {
                                status = customMatmulRun(ltHandle, operationDesc, alpha, // host or device pointer
                                    A, Adesc, B, Bdesc, beta,                            // host or device pointer
                                    C, Cdesc, C, Cdesc, algo, workSpace, workSpaceSize, perfResults[algoCount], stream,
                                    startEvent, stopEvent);
                                perfResults[algoCount].status = status;
                                if (status == CUBLAS_STATUS_SUCCESS)
                                {
                                    algoCount++;
                                }
                            }
                        }
                    } // end l
                }     // end k
            }         // end customOption
        }             // end tileIdx
        delete[] tileA;
    } // end idx

    printf("sort\n");

    // Sort the results per run duration
    std::sort(perfResults.begin(), perfResults.end(), timeCompare);

    // Print timing and perf details of the fastest combinations
    for (int32_t i = 0; i < kPRINT_ALGOS && i < algoCount; i++)
    {
        if (perfResults[i].time == customMatmulPerf_t::kMAX_TIME)
        {
            break;
        }
        printPerfStructure(perfResults[i], m, n, k);
    }

    // Descriptors are no longer needed as all GPU work was already enqueued
    PLUGIN_CUBLASASSERT(cublasLtMatmulPreferenceDestroy(preference));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Cdesc));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Bdesc));
    PLUGIN_CUBLASASSERT(cublasLtMatrixLayoutDestroy(Adesc));
    PLUGIN_CUBLASASSERT(cublasLtMatmulDescDestroy(operationDesc));
    PLUGIN_CUASSERT(cudaEventDestroy(startEvent));
    PLUGIN_CUASSERT(cudaEventDestroy(stopEvent));
}

class FusedMatmulBiasKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  FusedMatmulBiasKernel() = default;
  ~FusedMatmulBiasKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateCublasFusedMLPKernelCache();
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    const auto* matmul_cache = CHECK_NOTNULL(dynamic_cast<const CublasFusedMLPKernelCache*>(cache));

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::Tensor* _add_to_output = (ctx->has_input("_add_to_output", 0))
                                                ? ctx->Tensor4ArgNameAndIndex("_add_to_output", 0)
                                                : nullptr;

    const DataType data_type = out->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);
    size_t cublas_m = 0, cublas_n = 0, cublas_k = 0;
    int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0;

    const double alpha = ctx->Attr<double>("alpha");
    const double beta = (ctx->has_input("_add_to_output", 0)) ? ctx->Attr<double>("beta") : 0.0;

    const auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    const auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    DimVector in_shape({x->shape_view().Count(0, x->shape_view().NumAxes() - 1),
                        x->shape_view().At(x->shape_view().NumAxes() - 1)});

    DimVector weight_shape(2);

    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);

    weight->shape_view().ToDimVector(&weight_shape);

    InferMatmulCublasMNK(in_shape, weight_shape,
                         /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                         /*transpose_b=*/ep::primitive::BlasTransposeType::T, &cublas_m, &cublas_n,
                         &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    void* y_ptr = ctx->Tensor4ArgNameAndIndex("out", 0)->mut_dptr();

    SetCublasAttr(matmul_cache, cublas_compute_dtype, cuda_data_type, false,
                  /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                  /*transpose_b=*/ep::primitive::BlasTransposeType::T, epilogue, bias->dptr(),
                  nullptr, cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb, cublas_ldc);

    cublasLtMatmulPreference_t preference = nullptr;
    size_t workspace_size = cuda_stream->cublas_workspace_size();
    OF_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    OF_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
                                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &workspace_size, sizeof(workspace_size)));
    int returned_results = 0;
    cublasOperation_t transa, transb;
    std::vector<customMatmulPerf_t> perfResults(kNB_ALGO_COMBINATIONS);
    cublasLtMatmulDescGetAttribute(matmul_cache->operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t), nullptr);
    cublasLtMatmulDescGetAttribute(matmul_cache->operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t), nullptr);
    LtGemmSearch(cuda_stream->cublas_lt_handle(), transa, transb, cublas_m, cublas_n, cublas_k, &sp_alpha, weight->dptr(),
      cublas_lda, 
      x->dptr(), 
      cublas_ldb, 
      &sp_beta, 
      y_ptr, 
      cublas_ldc,
      cuda_stream->cublas_workspace(), 
      cuda_stream->cublas_workspace_size(),
      CUBLAS_COMPUTE_32F,
      CUDA_R_32F,
      GetCudaDataType(weight->data_type()),
      GetCudaDataType(x->data_type()),
      GetCudaDataType(out->data_type()),
      perfResults
      );

    
    auto HeuristicMatmul = [=]() -> void {
      constexpr int heuristic_num = 8;
      cublasLtMatmulHeuristicResult_t heuristic_result[heuristic_num];
      int returned_results = 0;
      OF_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
            cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, matmul_cache->cublas_a_desc,
            matmul_cache->cublas_b_desc, matmul_cache->cublas_c_desc, matmul_cache->cublas_c_desc,
            preference, heuristic_num, heuristic_result, &returned_results));

      cudaEvent_t st, ed;
      cudaEventCreate(&st, cudaEventBlockingSync);
      cudaEventCreate(&ed, cudaEventBlockingSync);
      constexpr int epochs = 128;
      std::vector<std::pair<int, float>> algoPairs;
      for (int i = 0; i < returned_results; i++) {

        for (int j = 0; j < epochs; j++) {
          OF_CUBLAS_CHECK(cublasLtMatmul(
            cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, &sp_alpha, weight->dptr(),
            matmul_cache->cublas_a_desc, x->dptr(), matmul_cache->cublas_b_desc, &sp_beta,
            (_add_to_output == nullptr) ? y_ptr : _add_to_output->dptr(), matmul_cache->cublas_c_desc,
            y_ptr, matmul_cache->cublas_c_desc, &heuristic_result[i].algo, cuda_stream->cublas_workspace(),
            cuda_stream->cublas_workspace_size(), cuda_stream->cuda_stream()));
        }
        cudaEventRecord(st, cuda_stream->cuda_stream());
        for (int j = 0; j < epochs; j++) {
          OF_CUBLAS_CHECK(cublasLtMatmul(
            cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, &sp_alpha, weight->dptr(),
            matmul_cache->cublas_a_desc, x->dptr(), matmul_cache->cublas_b_desc, &sp_beta,
            (_add_to_output == nullptr) ? y_ptr : _add_to_output->dptr(), matmul_cache->cublas_c_desc,
            y_ptr, matmul_cache->cublas_c_desc, &heuristic_result[i].algo, cuda_stream->cublas_workspace(),
            cuda_stream->cublas_workspace_size(), cuda_stream->cuda_stream()));
        }
        cudaEventRecord(ed, cuda_stream->cuda_stream());
        cudaEventSynchronize(st);
        cudaEventSynchronize(ed);

        float ms;
        cudaEventElapsedTime(&ms, st, ed);

        algoPairs.push_back(std::pair<int, float>(i, ms / epochs));
      }

      std::sort(algoPairs.begin(), algoPairs.end(), [](std::pair<int, float> pair1, std::pair<int, float> pair2) {return pair1.second < pair2.second;});

      for (int i = 0; i < returned_results; i++) {
        printf("algo_id %d cost %.4f ms\n", algoPairs[i].first, algoPairs[i].second);
      }

      return;
    };

    HeuristicMatmul();

    cublasLtMatmulHeuristicResult_t heuristic_result;
    OF_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, matmul_cache->cublas_a_desc,
        matmul_cache->cublas_b_desc, matmul_cache->cublas_c_desc, matmul_cache->cublas_c_desc,
        preference, 1, &heuristic_result, &returned_results));
    CHECK_EQ(returned_results, 1);
    cublasLtMatmulPreferenceDestroy(preference);
    OF_CUBLAS_CHECK(cublasLtMatmul(
        cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, &sp_alpha, weight->dptr(),
        matmul_cache->cublas_a_desc, x->dptr(), matmul_cache->cublas_b_desc, &sp_beta,
        (_add_to_output == nullptr) ? y_ptr : _add_to_output->dptr(), matmul_cache->cublas_c_desc,
        y_ptr, matmul_cache->cublas_c_desc, &heuristic_result.algo, cuda_stream->cublas_workspace(),
        cuda_stream->cublas_workspace_size(), cuda_stream->cuda_stream()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(data_type)               \
  REGISTER_USER_KERNEL("fused_matmul_bias")                            \
      .SetCreateFn<FusedMatmulBiasKernel>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == data_type));

REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(DataType::kDouble);
REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(DataType::kFloat);
REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(DataType::kFloat16);
#if CUDA_VERSION >= 11000
REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(DataType::kBFloat16);
#endif  // CUDA_VERSION >= 11000

}  // namespace

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11020
