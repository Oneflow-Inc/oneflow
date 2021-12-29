
def OneFlow_DecodeRandomOp : OneFlow_BaseOp<"decode_random", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
      Optional<OneFlow_Tensor>:$tick
  );
  let output = (outs
      OneFlow_Tensor:$out
  );
  let attrs = (ins
      ShapeAttr:$shape,
      DataType:$data_type,
      Optional<F32Attr>:$data_initializer_constant_value,
      Optional<SI64Attr>:$data_initializer_constant_int_value,
      Optional<F32Attr>:$data_initializer_random_uniform_min,
      Optional<F32Attr>:$data_initializer_random_uniform_max,
      Optional<SI32Attr>:$data_initializer_random_uniform_int_min,
      Optional<SI32Attr>:$data_initializer_random_uniform_int_max,
      Optional<F32Attr>:$data_initializer_random_normal_mean,
      Optional<F32Attr>:$data_initializer_random_normal_std,
      Optional<F32Attr>:$data_initializer_truncated_normal_mean,
      Optional<F32Attr>:$data_initializer_truncated_normal_std,
      Optional<UndefinedEnumVarianceNorm>:$data_initializer_xavier_variance_norm,
      Optional<StrAttr>:$data_initializer_xavier_data_format,
      Optional<UndefinedEnumVarianceNorm>:$data_initializer_msra_variance_norm,
      Optional<StrAttr>:$data_initializer_msra_data_format,
      Optional<F64Attr>:$data_initializer_range_start,
      Optional<F64Attr>:$data_initializer_range_stride,
      Optional<SI64Attr>:$data_initializer_range_axis,
      Optional<SI64Attr>:$data_initializer_int_range_start,
      Optional<SI64Attr>:$data_initializer_int_range_stride,
      Optional<SI64Attr>:$data_initializer_int_range_axis,
      Optional<F32Attr>:$data_initializer_variance_scaling_scale,
      Optional<UndefinedEnumVarianceNorm>:$data_initializer_variance_scaling_variance_norm,
      Optional<UndefinedEnumRandomDistribution>:$data_initializer_variance_scaling_distribution,
      Optional<StrAttr>:$data_initializer_variance_scaling_data_format,
      SI64Attr:$batch_size
  );
}

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
      Optional<SI32Attr>:$concat_box_axis,
      Optional<SI32Attr>:$split_box_axis,
      Optional<SI64ArrayAttr>:$split_box_part_num
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
      Optional<F32Attr>:$initializer_constant_value,
      Optional<SI64Attr>:$initializer_constant_int_value,
      Optional<F32Attr>:$initializer_random_uniform_min,
      Optional<F32Attr>:$initializer_random_uniform_max,
      Optional<SI32Attr>:$initializer_random_uniform_int_min,
      Optional<SI32Attr>:$initializer_random_uniform_int_max,
      Optional<F32Attr>:$initializer_random_normal_mean,
      Optional<F32Attr>:$initializer_random_normal_std,
      Optional<F32Attr>:$initializer_truncated_normal_mean,
      Optional<F32Attr>:$initializer_truncated_normal_std,
      Optional<UndefinedEnumVarianceNorm>:$initializer_xavier_variance_norm,
      Optional<StrAttr>:$initializer_xavier_data_format,
      Optional<UndefinedEnumVarianceNorm>:$initializer_msra_variance_norm,
      Optional<StrAttr>:$initializer_msra_data_format,
      Optional<F64Attr>:$initializer_range_start,
      Optional<F64Attr>:$initializer_range_stride,
      Optional<SI64Attr>:$initializer_range_axis,
      Optional<SI64Attr>:$initializer_int_range_start,
      Optional<SI64Attr>:$initializer_int_range_stride,
      Optional<SI64Attr>:$initializer_int_range_axis,
      Optional<F32Attr>:$initializer_variance_scaling_scale,
      Optional<UndefinedEnumVarianceNorm>:$initializer_variance_scaling_variance_norm,
      Optional<UndefinedEnumRandomDistribution>:$initializer_variance_scaling_distribution,
      Optional<StrAttr>:$initializer_variance_scaling_data_format,
      Optional<StrAttr>:$initialize_with_snapshot_path,
      Optional<StrAttr>:$initialize_with_snapshot_key,
      Optional<StrAttr>:$model_name,
      Optional<SI64Attr>:$random_seed,
      Optional<F32Attr>:$regularizer_l1_l2_l1,
      Optional<F32Attr>:$regularizer_l1_l2_l2,
      Optional<BoolAttr>:$trainable,
      Optional<StrArrayAttr>:$nd_sbp
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
