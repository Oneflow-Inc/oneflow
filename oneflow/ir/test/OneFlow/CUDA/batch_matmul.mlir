// RUN: oneflow-opt %s -lower-oneflow-to-tosa -tosa-to-linalg -cse --linalg-fuse-elementwise-ops -linalg-bufferize -convert-linalg-to-parallel-loops -gpu-greedy-parallel-loop-mapping \
// RUN: -convert-parallel-loops-to-gpu -gpu-kernel-outlining -buffer-host-register -canonicalize \
// RUN: -pass-pipeline='gpu.module(strip-debuginfo,lower-affine,convert-gpu-to-nvvm,out-of-tree-gpu-to-cubin)' \
// RUN: --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN: --std-bufferize --finalizing-bufferize \
// RUN: --convert-memref-to-llvm --convert-std-to-llvm \
// RUN: -gpu-to-llvm --reconcile-unrealized-casts \
// RUN: | tee %test_exec_root/$(basename %s).lower.mlir \
// RUN: | oneflow-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

