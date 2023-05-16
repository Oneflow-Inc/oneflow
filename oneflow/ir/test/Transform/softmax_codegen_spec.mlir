// RUN: oneflow-opt %s

transform.sequence failures(propagate) {
^bb1(%func_op: !pdl.operation):
  // Note: step 1, tiling and fusing linalg ops in block level.
  %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]}
    in %func_op : (!pdl.operation) -> !pdl.operation

  %match_0, %match_1, %match_2, %match_3, %match_end = transform.split_handle %ops
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation,
                           !pdl.operation, !pdl.operation)

  %forall, %_ =
    transform.structured.tile_to_forall_op %match_end tile_sizes [1, 4]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] )

  transform.structured.fuse_into_containing_op %match_3 into %forall
  transform.structured.fuse_into_containing_op %match_2 into %forall
  transform.structured.fuse_into_containing_op %match_1 into %forall
  transform.structured.fuse_into_containing_op %match_0 into %forall

  transform.oneflow.apply_patterns %func_op { canonicalization, cse } : (!pdl.operation) -> ()

  // Note: step 2, tiling and fusing linalg ops in thread level.
  %ops_1 = transform.structured.match ops{["linalg.fill", "linalg.generic"]}
    in %func_op : (!pdl.operation) -> !pdl.operation
  %match_0_0, %match_0_1, %match_0_2, %match_0_3, %match_0_end = transform.split_handle %ops_1
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation,
                           !pdl.operation, !pdl.operation)

  %reduction_linalg_ops = transform.merge_handles %match_0_1,
                                                  %match_0_3
    : !pdl.operation
  transform.structured.tile_to_forall_op %reduction_linalg_ops tile_sizes [1, 1]
    ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )

  %parallel_linalg_ops = transform.merge_handles %match_0_0,
                                                 %match_0_2,
                                                 %match_0_end
    : !pdl.operation
  transform.structured.tile_to_forall_op %parallel_linalg_ops num_threads [1, 4, 32]
    ( mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>] )

  // Note: step 3, vectorize + bufferize
  transform.oneflow.apply_patterns %func_op { canonicalization, cse } : (!pdl.operation) -> ()
  %func = transform.structured.match ops{["func.func"]} in %func_op : (!pdl.operation) -> !pdl.operation
  transform.structured.vectorize %func

  transform.bufferization.eliminate_empty_tensors %func_op

  %empty = transform.structured.match ops{["tensor.empty"]} in %func_op : (!pdl.operation) -> !pdl.operation
  %empty_id = transform.cast %empty : !pdl.operation to !transform.op<"tensor.empty">
  transform.bufferization.empty_tensor_to_alloc_tensor %empty_id : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
  transform.structured.hoist_redundant_tensor_subsets %func_op : (!pdl.operation) -> ()

  %bufferized_func_op = transform.oneflow.one_shot_bufferize  %func_op 
      {bufferize_function_boundaries = true,  allow_return_allocs = true, support_gpu = true } : (!pdl.operation) -> !pdl.operation

  // %bufferized_func_op = transform.bufferization.one_shot_bufferize %func_op
  //     {create_deallocs = false, bufferize_function_boundaries = true,  allow_return_allocs = true} : (!pdl.operation) -> !pdl.operation
  // transform.oneflow.apply_patterns %bufferized_func_op { cse } : (!pdl.operation) -> ()
  // transform.oneflow.apply_patterns %bufferized_func_op { memref_canonicalization } : (!pdl.operation) -> ()
  transform.oneflow.apply_patterns %bufferized_func_op { canonicalization, cse } : (!pdl.operation) -> ()

  // // // Note: step 4, mapping scf to gpu
  // %gpu_launch_op = transform.gpu.map_forall_to_blocks %bufferized_func_op { generate_gpu_launch }
  // transform.gpu.map_nested_forall_to_threads %gpu_launch_op block_dims = [32, 4, 1]


}

