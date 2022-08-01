
def OneFlow_CopyH2DOp : OneFlow_BaseOp<"copy_h2d", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
}

def OneFlow_CopyD2HOp : OneFlow_BaseOp<"copy_d2h", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
}

def OneFlow_CopyCommNetOp : OneFlow_BaseOp<"copy_comm_net", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
}

def OneFlow_BoxingOp : OneFlow_BaseOp<"boxing", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      SI32Attr:$in_num,
      SI32Attr:$out_num,
      Optional<SI32Attr>:$in_box__concat_box_axis,
      Optional<SI32Attr>:$out_box__split_box_axis,
      Optional<SI64ArrayAttr>:$out_box__split_box_part_num
  );
}

def OneFlow_VariableOp : OneFlow_BaseOp<"variable", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      Optional<OneFlow_Tensor>:$tick
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      ShapeAttr:$shape,
      Optional<DataType>:$data_type,
      Optional<F32Attr>:$type__initialize__initializer_constant_value,
      Optional<SI64Attr>:$type__initialize__initializer_constant_int_value,
      Optional<F32Attr>:$type__initialize__initializer_random_uniform_min,
      Optional<F32Attr>:$type__initialize__initializer_random_uniform_max,
      Optional<SI32Attr>:$type__initialize__initializer_random_uniform_int_min,
      Optional<SI32Attr>:$type__initialize__initializer_random_uniform_int_max,
      Optional<F32Attr>:$type__initialize__initializer_random_normal_mean,
      Optional<F32Attr>:$type__initialize__initializer_random_normal_std,
      Optional<F32Attr>:$type__initialize__initializer_trunc_normal_norm_mean,
      Optional<F32Attr>:$type__initialize__initializer_trunc_normal_norm_std,
      Optional<F32Attr>:$type__initialize__initializer_trunc_normal_min,
      Optional<F32Attr>:$type__initialize__initializer_trunc_normal_max,
      Optional<F64Attr>:$type__initialize__initializer_range_start,
      Optional<F64Attr>:$type__initialize__initializer_range_stride,
      Optional<SI64Attr>:$type__initialize__initializer_range_axis,
      Optional<SI64Attr>:$type__initialize__initializer_int_range_start,
      Optional<SI64Attr>:$type__initialize__initializer_int_range_stride,
      Optional<SI64Attr>:$type__initialize__initializer_int_range_axis,
      Optional<StrAttr>:$initialize__initialize_with_snapshot_path,
      Optional<StrAttr>:$initialize__initialize_with_snapshot_key,
      Optional<StrAttr>:$model_name,
      Optional<SI64Attr>:$random_seed,
      Optional<F32Attr>:$type__regularizer_l1_l2_l1,
      Optional<F32Attr>:$type__regularizer_l1_l2_l2,
      Optional<BoolAttr>:$trainable,
      StrArrayAttr:$nd_sbp
  );
}

def OneFlow_TickOp : OneFlow_BaseOp<"tick", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$tick
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
}

def OneFlow_CriticalSectionWaitTickOp : OneFlow_BaseOp<"critical_section_wait_tick", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$tick
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrAttr:$buffer_name
  );
}

def OneFlow_CriticalSectionCallbackTickOp : OneFlow_BaseOp<"critical_section_callback_tick", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$tick
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrAttr:$buffer_name
  );
}

def OneFlow_TotalLossInstanceNumOp : OneFlow_BaseOp<"total_loss_instance_num", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
}

def OneFlow_ShapeElemCntOp : OneFlow_BaseOp<"shape_elem_cnt", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$x
  );
  let output = (outs
      OneFlow_Tensor:$y
  );
  let attrs = (ins
      Optional<DataType>:$data_type
  );
}

def OneFlow_SrcSubsetTickOp : OneFlow_BaseOp<"src_subset_tick", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
}

def OneFlow_DstSubsetTickOp : OneFlow_BaseOp<"dst_subset_tick", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
}

def OneFlow_SourceTickOp : OneFlow_BaseOp<"source_tick", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
      OneFlow_Tensor:$out
  );
}

def OneFlow_SinkTickOp : OneFlow_BaseOp<"sink_tick", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$tick
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
}

def OneFlow_InputOp : OneFlow_BaseOp<"input", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      Optional<OneFlow_Tensor>:$tick
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      Optional<ShapeAttr>:$blob_shape,
      Optional<DataType>:$blob_data_type,
      Optional<BoolAttr>:$blob_is_dynamic,
      Optional<StrArrayAttr>:$blob_nd_sbp,
      Optional<StrAttr>:$job_name
  );
}

def OneFlow_OutputOp : OneFlow_BaseOp<"output", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      Optional<ShapeAttr>:$blob_shape,
      Optional<DataType>:$blob_data_type,
      Optional<BoolAttr>:$blob_is_dynamic,
      Optional<StrArrayAttr>:$blob_nd_sbp,
      Optional<StrAttr>:$job_name
  );
}

def OneFlow_WaitAndSendIdsOp : OneFlow_BaseOp<"wait_and_send_ids", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrAttr:$wait_buffer_name,
      SI64ArrayAttr:$id_list,
      DataType:$data_type,
      Optional<StrAttr>:$job_name
  );
}

def OneFlow_ReentrantLockOp : OneFlow_BaseOp<"reentrant_lock", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrAttr:$start,
      Optional<StrAttr>:$end,
      SI64ArrayAttr:$lock_id2intersecting_lock_ids
  );
}

def OneFlow_CallbackNotifyOp : OneFlow_BaseOp<"callback_notify", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrArrayAttr:$callback_buffer_name,
      Optional<StrAttr>:$job_name
  );
}

def OneFlow_ForeignInputOp : OneFlow_BaseOp<"foreign_input", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      Optional<OneFlow_Tensor>:$tick
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      Optional<ShapeAttr>:$blob_shape,
      Optional<DataType>:$blob_data_type,
      Optional<BoolAttr>:$blob_is_dynamic,
      Optional<StrArrayAttr>:$blob_nd_sbp,
      StrAttr:$ofblob_buffer_name
  );
}

def OneFlow_ForeignOutputOp : OneFlow_BaseOp<"foreign_output", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrAttr:$ofblob_buffer_name
  );
}

def OneFlow_AccTickOp : OneFlow_BaseOp<"acc_tick", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrAttr:$one,
      StrAttr:$acc,
      Optional<SI32Attr>:$max_acc_num
  );
}

def OneFlow_ReturnOp : OneFlow_BaseOp<"return", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      Optional<StrAttr>:$job_name
  );
}

def OneFlow_ForeignWatchOp : OneFlow_BaseOp<"foreign_watch", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrAttr:$handler_uuid
  );
}

def OneFlow_DistributeConcatOp : OneFlow_BaseOp<"distribute_concat", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      SI32Attr:$axis
  );
}

def OneFlow_DistributeSplitOp : OneFlow_BaseOp<"distribute_split", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      SI32Attr:$axis,
      Optional<BoolAttr>:$is_variable_ref
  );
}

def OneFlow_DistributeCloneOp : OneFlow_BaseOp<"distribute_clone", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      Optional<BoolAttr>:$is_variable_ref
  );
}

def OneFlow_DistributeAddOp : OneFlow_BaseOp<"distribute_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
}

def OneFlow_DeviceTickOp : OneFlow_BaseOp<"device_tick", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$tick
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      Optional<ShapeAttr>:$time_shape
  );
}

def OneFlow_SliceBoxingCopyOp : OneFlow_BaseOp<"slice_boxing_copy", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$slice_boxing_in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      SI64Attr:$slice_boxing_in_slice_dim_begin,
      SI64Attr:$slice_boxing_in_slice_dim_end,
      SI64Attr:$slice_boxing_out_slice_dim_begin,
      SI64Attr:$slice_boxing_out_slice_dim_end,
      Optional<ShapeAttr>:$slice_boxing_out_shape
  );
}

def OneFlow_SliceBoxingAddOp : OneFlow_BaseOp<"slice_boxing_add", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$slice_boxing_in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      SI64Attr:$slice_boxing_in_slice_dim_begin,
      SI64Attr:$slice_boxing_in_slice_dim_end,
      SI64Attr:$slice_boxing_out_slice_dim_begin,
      SI64Attr:$slice_boxing_out_slice_dim_end,
      Optional<ShapeAttr>:$slice_boxing_out_shape
  );
}

def OneFlow_CollectiveBoxingGenericOp : OneFlow_BaseOp<"collective_boxing_generic", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrAttr:$rank_desc_op_desc_name,
      UndefinedEnumOpType:$rank_desc_op_desc_op_type,
      Optional<UndefinedEnumReduceMethod>:$rank_desc_op_desc_reduce_method,
      Optional<SI64Attr>:$rank_desc_op_desc_root,
      DataType:$rank_desc_op_desc_data_type,
      ShapeAttr:$rank_desc_op_desc_shape,
      SI64Attr:$rank_desc_op_desc_num_ranks,
      UndefinedEnumBackend:$rank_desc_op_desc_backend,
      SI64Attr:$rank_desc_rank
  );
}

def OneFlow_BoxingIdentityOp : OneFlow_BaseOp<"boxing_identity", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
}

def OneFlow_CollectiveBoxingPackOp : OneFlow_BaseOp<"collective_boxing_pack", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrAttr:$src_sbp,
      StrAttr:$dst_sbp,
      SI64Attr:$num_ranks,
      ShapeAttr:$logical_shape
  );
}

def OneFlow_CollectiveBoxingUnpackOp : OneFlow_BaseOp<"collective_boxing_unpack", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrAttr:$src_sbp,
      StrAttr:$dst_sbp,
      SI64Attr:$num_ranks,
      ShapeAttr:$logical_shape
  );
}

def OneFlow_BoxingZerosOp : OneFlow_BaseOp<"boxing_zeros", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      ShapeAttr:$shape,
      DataType:$data_type
  );
}

def OneFlow_NcclSendRecvBoxingOp : OneFlow_BaseOp<"nccl_send_recv_boxing", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      OneFlow_Tensor:$in
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      StrArrayAttr:$src_nd_sbp,
      StrArrayAttr:$dst_nd_sbp,
      StrArrayAttr:$parallel_device_name,
      StrAttr:$parallel_device_tag,
      Optional<ShapeAttr>:$parallel_hierarchy,
      StrArrayAttr:$src_parallel_device_name,
      StrAttr:$src_parallel_device_tag,
      Optional<ShapeAttr>:$src_parallel_hierarchy,
      StrArrayAttr:$dst_parallel_device_name,
      StrAttr:$dst_parallel_device_tag,
      Optional<ShapeAttr>:$dst_parallel_hierarchy,
      ShapeAttr:$logical_shape,
      DataType:$data_type,
      BoolAttr:$has_input,
      BoolAttr:$has_output
  );
}
