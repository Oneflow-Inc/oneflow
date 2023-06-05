// RUN: oneflow-opt %s

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  // Note: step 1, tiling and fusing linalg ops in block level.
  %ops = transform.structured.match ops{["linalg.fill", "linalg.generic"]}
    in %module_op : (!pdl.operation) -> !pdl.operation

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

  transform.oneflow.canonicalization %module_op : (!pdl.operation) -> ()
  transform.oneflow.cse %module_op : (!pdl.operation) -> ()


  // Note: step 2, tiling and fusing linalg ops in thread level.
  %ops_1 = transform.structured.match ops{["linalg.fill", "linalg.generic"]}
    in %module_op : (!pdl.operation) -> !pdl.operation
  %match_0_0,
  %match_0_1,
  %match_0_2,
  %match_0_3,
  %match_0_end = transform.split_handle %ops_1
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
  transform.oneflow.canonicalization %module_op : (!pdl.operation) -> ()
  transform.oneflow.cse %module_op : (!pdl.operation) -> ()

  // Note: step 3, bufferize
  transform.oneflow.explicit_linalg_outcome %module_op : (!pdl.operation) -> ()

  transform.bufferization.eliminate_empty_tensors %module_op

  %empty = transform.structured.match ops{["tensor.empty"]} in %module_op : (!pdl.operation) -> !pdl.operation
  %empty_id = transform.cast %empty : !pdl.operation to !transform.op<"tensor.empty">
  transform.bufferization.empty_tensor_to_alloc_tensor %empty_id : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">

  %bufferized_module_op = transform.bufferization.one_shot_bufferize %module_op
      {create_deallocs = false, bufferize_function_boundaries = true,  allow_return_allocs = true} : (!pdl.operation) -> !pdl.operation
      
  // Note: step 4, post bufferize function-type-related transform
  transform.oneflow.canonicalization %bufferized_module_op : (!pdl.operation) -> ()
  transform.oneflow.cse %bufferized_module_op : (!pdl.operation) -> ()
  transform.oneflow.eliminate_copy %bufferized_module_op : (!pdl.operation) -> ()

  %func = transform.structured.match ops{["func.func"]} in %bufferized_module_op : (!pdl.operation) -> !pdl.operation
  transform.structured.hoist_redundant_tensor_subsets %func
    : (!pdl.operation) -> ()

  // Note: step 5, post bufferize memory-buffer-pool transform
  transform.oneflow.results_to_out_params %bufferized_module_op : (!pdl.operation) -> ()
  transform.oneflow.eliminate_copy %bufferized_module_op : (!pdl.operation) -> ()
  transform.oneflow.fold_alloc %func : (!pdl.operation) -> ()

  // Note: step 6, mapping scf to gpu
  %gpu_launch_op = transform.gpu.map_forall_to_blocks %bufferized_module_op { generate_gpu_launch }
  transform.gpu.map_nested_forall_to_threads %gpu_launch_op block_dims = [32, 4, 1]
}

