// RUN: oneflow-opt %s

transform.sequence failures(propagate) {
^bb1(%func_op: !pdl.operation):
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

  transform.oneflow.apply_patterns %func_op { canonicalization } : (!pdl.operation) -> ()
  transform.oneflow.apply_patterns %func_op { cse } : (!pdl.operation) -> ()

  // transform.print %func_op {name = "after tiling and fusing in thread level - softmax v2"}: !pdl.operation

  // %ops_1 = transform.structured.match ops{["linalg.fill", "linalg.generic"]}
  //   in %func_op : (!pdl.operation) -> !pdl.operation
  // %match_0_0,
  // %match_0_1,
  // %match_0_2,
  // %match_0_3,
  // %match_0_end = transform.split_handle %ops_1
  //   : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation,
  //                          !pdl.operation, !pdl.operation)

  // %reduction_linalg_ops = transform.merge_handles %match_0_1,
  //                                                 %match_0_3
  //   : !pdl.operation
  // transform.structured.tile_to_forall_op %reduction_linalg_ops tile_sizes [1, 1]

  //   ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )
  // %parallel_linalg_ops = transform.merge_handles %match_0_0,
  //                                                %match_0_2,
  //                                                %match_0_end
  //   : !pdl.operation
  // transform.structured.tile_to_forall_op %parallel_linalg_ops num_threads [1, 4, 32]
  //   ( mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>] )

  // transform.oneflow.apply_patterns %func_op { canonicalization } : (!pdl.operation) -> ()

  // transform.print %func_op {name = "after tiling and fusing in thread level - softmax v2"}: !pdl.operation


//   %gpuLaunch = transform.gpu.map_forall_to_blocks %funcop { generate_gpu_launch }
//   transform.gpu.map_nested_forall_to_threads %gpuLaunch block_dims = [32, 4, 1]


}

