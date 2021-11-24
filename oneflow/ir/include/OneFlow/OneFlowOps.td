#ifndef ONEFLOW_OPS
#define ONEFLOW_OPS

include "OneFlowDialect.td"
include "OneFlowEnums.td"
include "OneFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Pass/PassBase.td"
include "mlir/Interfaces/CallInterfaces.td"

def SI32ArrayAttr : TypedArrayAttrBase<SI32Attr,
                                      "signed 32-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getArrayAttr(llvm::to_vector<8>(llvm::map_range(values, [this](int32_t v) -> Attribute { return builder_.getSI32IntegerAttr($0); })))";
}

def SI64ArrayAttr : TypedArrayAttrBase<SI64Attr,
                                      "signed 64-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getArrayAttr(llvm::to_vector<8>(llvm::map_range(values, [this](int64_t v) -> Attribute { return builder_.getSI64IntegerAttr($0); })))";
}

def DTArrayAttr : TypedArrayAttrBase<OneFlow_DataType,
                                      "signed 64-bit integer array attribute"> {
  let constBuilderCall = "$_builder.getArrayAttr(llvm::to_vector<8>(llvm::map_range(values, [this](auto v) -> Attribute { return DataTypeAttr::get($0); })))";
}

def ShapeArrayAttr : TypedArrayAttrBase<AnyI64ElementsAttr,
                                      ""> {
  let constBuilderCall = "$_builder.getArrayAttr(llvm::to_vector<8>(llvm::map_range(values, [this](auto v) -> Attribute { return DenseIntElementsAttr::get($0); })))";
}

class OneFlow_BaseOp<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, traits> {
  dag sys_attrs = (ins
    StrAttr:$op_name,
    OptionalAttr<BoolAttr>:$trainable,
    StrAttr:$device_tag,
    StrArrayAttr:$device_name, // TODO: change device_name to dict and parse the literal fmt like "0:0-0"
    OptionalAttr<I64Attr>:$scope_symbol_id,
    OptionalAttr<I64ArrayAttr>:$hierarchy
  );
  dag attrs = (ins);
  dag trait_attrs = (ins);
  dag user_op_attrs = (ins);
  dag input = (ins Variadic<AnyType>:$data_input);
  dag output = (outs Variadic<AnyType>:$data_output);
  dag ctrl_input = (ins);
  dag ctrl_output = (outs);
  let arguments = !con(
      input,
      ctrl_input,
      sys_attrs,
      trait_attrs,
      user_op_attrs,
      attrs
  );
  let results = !con(
    output,
    ctrl_output
  );
}

class OneFlow_Op<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, AttrSizedResultSegments])> {
  let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
  let ctrl_output = (outs Optional<AnyType>:$ctrl_output);
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
}

class OneFlow_UserBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, traits> {
    let summary = "";
    let user_op_attrs = (ins
      StrAttr:$op_type_name,
      StrArrayAttr:$input_lbn_segment_keys,
      I32ArrayAttr:$input_lbn_segment_sizes,
      StrArrayAttr:$output_lbn_segment_keys,
      I32ArrayAttr:$output_lbn_segment_sizes,
      StrArrayAttr:$output_lbns
    );
}

// Why don't we merge ctrl in/out and data in/out into operand_segment/result_segment_sizes?
// 1. We only need to erase operand_segment/result_segment_sizes when we are creating a concrete user op
// 2. Isolating data and ctrl make debug easier and produced IR more human-readable
class OneFlow_UserBaseWithCtrlOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_UserBaseOp<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, AttrSizedResultSegments])> {
    let summary = "";
    let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
    let ctrl_output = (outs Optional<AnyType>:$ctrl_output);
    let trait_attrs = (ins
      I32ElementsAttr:$operand_segment_sizes,
      I32ElementsAttr:$result_segment_sizes
    );
}

def OneFlow_UserOp : OneFlow_UserBaseOp<"user", [AttrSizedOperandSegments, AttrSizedResultSegments]> {
    let summary = "";
    let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
    let ctrl_output = (outs Optional<AnyType>:$ctrl_output);
    let trait_attrs = (ins
      I32ElementsAttr:$operand_segment_sizes,
      I32ElementsAttr:$result_segment_sizes
    );
    let hasCanonicalizer = 1;
}

def OneFlow_SystemOp : OneFlow_Op<"system", []> {
  let summary = "";
  let attrs = (ins
    StrArrayAttr:$input_bns,
    StrArrayAttr:$output_lbns
  );
  let hasCanonicalizer = 1;
}

class OneFlow_ConvolutionBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let summary = "OneFlow convolution operation";
    let description = [{
    "The convolution operator consumes an input tensor and a filter, and"
    "computes the output."
    }];
    let input = (ins
      AnyType:$in,
      AnyType:$weight
    );
    let output = (outs AnyType:$out);
    let attrs = (ins
      SI32Attr:$filters,
      SI32ArrayAttr:$padding_before,
      StrAttr:$data_format,
      SI32ArrayAttr:$kernel_size,
      SI32ArrayAttr:$strides,
      SI32ArrayAttr:$dilation_rate,
      DefaultValuedAttr<SI32Attr, "1">:$group
    );
}

class OneFlow_LazyPoolBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let summary = "OneFlow Lazy Pooling operation";
    let input = (ins AnyType:$x);
    let output = (outs AnyType:$y);
    let attrs = (ins
    StrAttr:$padding,
    SI32ArrayAttr:$padding_before,
    SI32ArrayAttr:$padding_after,
    StrAttr:$data_format,
    SI32ArrayAttr:$pool_size,
    SI32ArrayAttr:$strides,
    BoolAttr:$ceil_mode
  );
}

class OneFlow_LazyPoolGradBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let summary = "OneFlow Lazy Pooling Grad operation";
    let input = (ins
      AnyType:$x,
      AnyType:$y,
      AnyType:$dy
    );
    let output = (outs AnyType:$dx);
    let attrs = (ins
    StrAttr:$padding,
    SI32ArrayAttr:$padding_before,
    SI32ArrayAttr:$padding_after,
    StrAttr:$data_format,
    SI32ArrayAttr:$pool_size,
    SI32ArrayAttr:$strides,
    BoolAttr:$ceil_mode
  );
}


class OneFlow_EagerPoolBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let summary = "OneFlow Eager Pooling operation";
    let input = (ins AnyType:$x);
    let output = (outs AnyType:$y);
    let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad,
    SI64Attr:$divisor_override
  );
}

class OneFlow_EagerPoolGradBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
    let summary = "OneFlow Eager Pooling Grad operation";
    let input = (ins
      AnyType:$x,
      AnyType:$y,
      AnyType:$dy
    );
    let output = (outs AnyType:$dx);
    let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    BoolAttr:$ceil_mode,
    BoolAttr:$count_include_pad,
    SI64Attr:$divisor_override
  );
}

class OneFlow_UnaryBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType, NoSideEffect])> {
    let summary = "";
    let input = (ins AnyType:$x);
    let output = (outs AnyType:$y);
}

def OneFlow_Idempotent : NativeOpTrait<"IsIdempotentOfIdenticalPlacement">;

class OneFlow_IdempotentBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_UnaryBaseOp<mnemonic, !listconcat(traits, [OneFlow_Idempotent])> {
}

def OneFlow_Involution : NativeOpTrait<"IsInvolutionOfIdenticalPlacement">;

class OneFlow_InvolutionBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_UnaryBaseOp<mnemonic, !listconcat(traits, [OneFlow_Involution])> {
}

include "OneFlowUserOpGen.td"

def OneFlow_NormalizationAddReluOp : OneFlow_NormalizationAddReluBaseOp {
  let builders = [
    OpBuilder<(ins
      "Value":$x,
      "Value":$addend,
      "Value":$moving_mean,
      "Value":$moving_variance,
      "Value":$gamma,
      "Value":$beta,
      "StringRef":$op_name,
      "BoolAttr":$trainable,
      "StringRef":$device_tag,
      "ArrayAttr":$device_name,
      "IntegerAttr":$scope_symbol_id,
      "ArrayAttr":$hierarchy,
      "DenseElementsAttr":$operand_segment_sizes,
      "DenseElementsAttr":$result_segment_sizes,
      "IntegerAttr":$axis,
      "FloatAttr":$epsilon,
      "BoolAttr":$training,
      "FloatAttr":$momentum
    )>
  ];
}

def OneFlow_Add2Op : OneFlow_BaseOp<"add_n2", [NoSideEffect, DeclareOpInterfaceMethods<BnOrderOpInterface>]> {
  let summary = "";
  let input = (ins
    AnyType:$in0,
    AnyType:$in1
  );
  let output = (outs AnyType:$out);
  let extraClassDeclaration = [{
    static std::vector<std::string> inputOrder() { return {"in"}; }
    static std::vector<std::string> outputOrder() { return {"out"}; }
  }];
}

// JIT ops

def OneFlow_MlirJitOp : OneFlow_UserBaseOp<"mlir_jit", [
      CallOpInterface
    ]
  > {
  let attrs = (ins
    FlatSymbolRefAttr:$callee,
    StrAttr:$mlir_assembly
  );
  let builders = [
    OpBuilder<(ins "FuncOp":$callee,
      "NamedAttrList":$attributes,
      CArg<"ValueRange", "{}">:$data_input), [{
      $_state.addOperands(data_input);
      $_state.addAttributes(attributes);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getType().getResults());
    }]>
  ];
  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];
  let assemblyFormat = [{
    $callee `(` $data_input `)` attr-dict `:` functional-type($data_input, results)
  }];
}

#endif // ONEFLOW_OPS

#ifndef ONEFLOW_PATTERNS
#define ONEFLOW_PATTERNS

def IsNotNestedInJit: Constraint<CPred<"(!$0.getDefiningOp()->getParentOfType<FuncOp>()->hasAttr(\"llvm.emit_c_interface\"))">, "">;
def OutlineMulCast : NativeCodeCall<"OutlineMulCast($_builder, $0, $1)">;
// TODO: remove attr binding if possible
def MulCastPattern : Pat<
  (
    OneFlow_ScalarMulByTensorOp : $mul_op
    (
      OneFlow_CastOp : $cast_op
        $cast_x,
        $cast_op_name,
        $cast_trainable,
        $cast_device_tag,
        $cast_device_name,
        $cast_scope_symbol_id,
        $cast_hierarchy,
        $cast_dtype
    ),
    $scalar,
    $mul_op_name,
    $mul_trainable,
    $mul_device_tag,
    $mul_device_name,
    $mul_scope_symbol_id,
    $mul_hierarchy
  ),
  (OutlineMulCast $mul_op, $cast_op),
  [
    (IsNotNestedInJit $mul_op)
  ]
>;

def GetFirstValue :
  NativeCodeCall<"*$0.begin()">;

def IsGPU: Constraint<CPred<"$0.getValue().equals(\"gpu\")">, "is GPU device">;

def FusedScaleTrilPattern : Pat<
  (
    OneFlow_TrilOp
    (
      OneFlow_ScalarMulOp
        $x,
        $scale_op_name,
        $scale_trainable,
        $scale_device_tag,
        $scale_device_name,
        $scale_scope_symbol_id,
        $scale_hierarchy,
        $has_int_operand,
        $has_float_operand,
        $int_operand,
        $float_operand
    ),
    $tril_op_name,
    $tril_trainable,
    $tril_device_tag,
    $tril_device_name,
    $tril_scope_symbol_id,
    $tril_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value
  ),
  (OneFlow_FusedScaleTrilOp $x,
    $tril_op_name,
    $tril_trainable,
    $tril_device_tag,
    $tril_device_name,
    $tril_scope_symbol_id,
    $tril_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value,
    $float_operand,
    $int_operand,
    $has_float_operand
  ),
  [
    (IsGPU $tril_device_tag),
    (IsGPU $scale_device_tag)
  ]
>;

// TODO: use either to merge two patterns
def FusedScaleTrilPattern2 : Pat<
  (
    OneFlow_ScalarMulOp
    (
      OneFlow_TrilOp
      $x,
      $tril_op_name,
      $tril_trainable,
      $tril_device_tag,
      $tril_device_name,
      $tril_scope_symbol_id,
      $tril_hierarchy,
      $diagonal,
      $floating_fill_value,
      $integer_fill_value,
      $is_floating_fill_value
    ),
    $scale_op_name,
    $scale_trainable,
    $scale_device_tag,
    $scale_device_name,
    $scale_scope_symbol_id,
    $scale_hierarchy,
    $has_int_operand,
    $has_float_operand,
    $int_operand,
    $float_operand
  ),
  (OneFlow_FusedScaleTrilOp $x,
    $scale_op_name,
    $scale_trainable,
    $scale_device_tag,
    $scale_device_name,
    $scale_scope_symbol_id,
    $scale_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value,
    $float_operand,
    $int_operand,
    $has_float_operand
  ),
  [
    (IsGPU $tril_device_tag),
    (IsGPU $scale_device_tag)
  ]
>;

def FusedBiasAddDropoutPattern : Pat<
  (
    OneFlow_DropoutOp
    (
      OneFlow_BiasAddOp
        $a,
        $b,
        $bias_add_op_name,
        $bias_add_trainable,
        $bias_add_device_tag,
        $bias_add_device_name,
        $bias_add_scope_symbol_id,
        $bias_add_hierarchy,
        $axis
    ),
    $mask,
    $_add_to_output,
    $dropout_op_name,
    $dropout_trainable,
    $dropout_device_tag,
    $dropout_device_name,
    $dropout_scope_symbol_id,
    $dropout_hierarchy,
    $scale
  ),
  (OneFlow_FusedBiasAddMaskScaleOp $a, $b, $mask, (GetFirstValue $_add_to_output),
    $dropout_op_name,
    $dropout_trainable,
    $dropout_device_tag,
    $dropout_device_name,
    $dropout_scope_symbol_id,
    $dropout_hierarchy,
    $axis,
    $scale
  ),
  []
>;

def FusedBiasAddGeluPattern : Pat<
  (
    OneFlow_GeluOp : $gelu_op
    (
      OneFlow_BiasAddOp
        $a,
        $b,
        $bias_add_op_name,
        $bias_add_trainable,
        $bias_add_device_tag,
        $bias_add_device_name,
        $bias_add_scope_symbol_id,
        $bias_add_hierarchy,
        $axis
    ),
    $gelu_op_name,
    $gelu_trainable,
    $gelu_device_tag,
    $gelu_device_name,
    $gelu_scope_symbol_id,
    $gelu_hierarchy
  ),
  (OneFlow_FusedBiasAddGeluOp $a, $b,
    $gelu_op_name,
    $gelu_trainable,
    $gelu_device_tag,
    $gelu_device_name,
    $gelu_scope_symbol_id,
    $gelu_hierarchy,
    $axis
  ),
  []
>;

def IsTraingTrue: Constraint<CPred<"$0.getValue()">, "">;
// TODO: check mean and inv_variance are not used
def NormalizationAddReluPattern : Pattern<
  /* match */ (
    OneFlow_ReluOp (
      OneFlow_Add2Op (
          OneFlow_NormalizationOp:$results
            $bn_x,
            $bn_moving_mean,
            $bn_moving_variance,
            $bn_gamma,
            $bn_beta,
            $bn__add_to_output, // TODO: check this is none
            $bn_op_name,
            $bn_trainable,
            $bn_device_tag,
            $bn_device_name,
            $bn_scope_symbol_id,
            $bn_hierarchy,
            $operand_segment_sizes,
            $result_segment_sizes,
            $bn_axis,
            $bn_epsilon,
            $bn_training,
            $bn_momentum
      ),
      $addend,
      $add_op_name,
      $add_trainable,
      $add_device_tag,
      $add_device_name,
      $add_scope_symbol_id,
      $add_hierarchy
    ),
    $relu_op_name,
    $relu_trainable,
    $relu_device_tag,
    $relu_device_name,
    $relu_scope_symbol_id,
    $relu_hierarchy
  ),
  /* replace */ [(
    OneFlow_NormalizationAddReluOp: $fuse_results
    $bn_x,
    $addend,
    (GetFirstValue $bn_moving_mean),
    (GetFirstValue $bn_moving_variance),
    $bn_gamma,
    $bn_beta,
    $bn_op_name,
    $bn_trainable,
    $bn_device_tag,
    $bn_device_name,
    $bn_scope_symbol_id,
    $bn_hierarchy,
    /* not used */ $operand_segment_sizes,
    /* not used */ $result_segment_sizes,
    $bn_axis,
    $bn_epsilon,
    $bn_training,
    $bn_momentum
  ),
  (replaceWithValue $fuse_results__0),
  ],
  [(IsTraingTrue $bn_training)]
>;

#endif // ONEFLOW_PATTERNS

#ifndef ONEFLOW_PASSES
#define ONEFLOW_PASSES

def LowerOneFlowToTosaPass : Pass<"lower-oneflow-to-tosa", "ModuleOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createLowerOneFlowToTosaPass()";
  let dependentDialects = ["tosa::TosaDialect", "memref::MemRefDialect", "StandardOpsDialect"];
}

def MapSCFToGPUPass : Pass<"gpu-greedy-parallel-loop-mapping", "FuncOp"> {
  let summary = "Greedily maps all parallel loops to gpu hardware ids";
  let constructor = "mlir::oneflow::createMapSCFToGPUPass()";
  let dependentDialects = ["scf::SCFDialect"];
}

def BufferHostRegisterPass : FunctionPass<"buffer-host-register"> {
  let summary = "";
  let constructor = "mlir::oneflow::createBufferHostRegisterPass()";
  let dependentDialects = ["gpu::GPUDialect"];
}

def OutlineJitFunctionPass : Pass<"outline-jit-function", "ModuleOp"> {
  let summary = "move ops could be jitted to jit function";
  let constructor = "mlir::oneflow::createOutlineJitFunctionPass()";
}

def FuseIntoExistingOpPass : Pass<"fuse-into-existing-op", "FuncOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createFuseIntoExistingOpPass()";
}

#endif // ONEFLOW_PASSES
