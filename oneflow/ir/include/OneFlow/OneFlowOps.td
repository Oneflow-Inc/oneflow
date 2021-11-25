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

#define GET_ONEFLOW_BASE_OP_DEFINITIONS
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
