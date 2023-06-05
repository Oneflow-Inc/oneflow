// RUN: oneflow-opt %s --pass-pipeline="builtin.module(oneflow-transform-dialect-interpreter{transform-file-name=%p/softmax_codegen_spec_no_vectorize.mlir})" \
// RUN: | oneflow-opt  --insert-ofmempool  --convert-linalg-to-loops --convert-scf-to-cf --canonicalize --cse --memref-expand  --gpu-kernel-outlining \
// RUN: | oneflow-opt --pass-pipeline='builtin.module(gpu.module(expand-strided-metadata,lower-affine,strip-debuginfo,convert-gpu-to-nvvm,nvvm-to-cubin))'


!tmp_tensor_t = tensor<16x128xf32>
!in_tensor_t = tensor<16x128x128xf32>
!out_tensor_t = tensor<16x128x128xf32>

func.func @softmax() -> !out_tensor_t {
  %cst_0 = arith.constant 0.0 : f32
  %cst_1 = arith.constant 1.0 : f32
  %cst_min = arith.constant -3.40282347E+38 : f32
  %input = arith.constant dense<5.000000e+00> : !out_tensor_t

  %input_max_empty = tensor.empty() : !tmp_tensor_t
  %input_max_filled = linalg.fill ins(%cst_min : f32)
    outs(%input_max_empty : !tmp_tensor_t) -> !tmp_tensor_t
  %input_max = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>],
                      iterator_types = ["parallel", "parallel", "reduction"]}
     ins(%input : !in_tensor_t)
    outs(%input_max_filled : !tmp_tensor_t) {
      ^bb0(%arg0: f32, %arg1: f32):
        %max = arith.maxf %arg0, %arg1 : f32
        linalg.yield %max : f32
      } -> !tmp_tensor_t

  // This has been fused manually to avoid the fusion on tensors pass and reduce noise atm.
  %exps_empty = tensor.empty() : !out_tensor_t
  %exps_sum_empty = tensor.empty() : !tmp_tensor_t
  %exps_sum_filled = linalg.fill ins(%cst_0 : f32)
    outs(%exps_sum_empty : !tmp_tensor_t) -> !tmp_tensor_t
  %exps, %exps_sum = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>],
                      iterator_types = ["parallel", "parallel", "reduction"]}
     ins(%input, %input_max : !in_tensor_t, !tmp_tensor_t)
    outs(%exps_empty, %exps_sum_filled : !out_tensor_t, !tmp_tensor_t) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
        %sub = arith.subf %arg0, %arg1 : f32
        %exp = math.exp %sub : f32
        %add = arith.addf %exp, %arg3 : f32
        linalg.yield %exp, %add : f32, f32
      } -> (!out_tensor_t, !tmp_tensor_t)

  %res_empty = tensor.empty() : !out_tensor_t
  %res = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                      iterator_types = ["parallel", "parallel", "parallel"]}
     ins(%exps, %exps_sum : !out_tensor_t, !tmp_tensor_t)
    outs(%res_empty : !out_tensor_t) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
        // %10 = arith.divf %cst_1, %arg1 : f32
        // %11 = arith.mulf %arg0, %10 : f32
        %div = arith.divf %arg0, %arg1 : f32
        linalg.yield %div : f32
      } -> !out_tensor_t

  return %res: !out_tensor_t
}
