#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWBASE_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWBASE_H_

include "OneFlow/OneFlowDialect.td"
include "OneFlow/OneFlowInterfaces.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

def OneFlow_IsOpConfCompatible : NativeOpTrait<"IsOpConfCompatible">;
def OneFlow_IsImportCompatible : NativeOpTrait<"IsImportCompatible">;
def OneFlow_AlternativeOp : NativeOpTrait<"IsAlternative">;

class OneFlow_BaseOp<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, !listconcat(traits, [OneFlow_IsOpConfCompatible])> {
  dag op_conf_attrs = (ins
    StrAttr:$op_name,
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
      op_conf_attrs,
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
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<ControlEdgeCompatibleInterface>])> {
  let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
  let ctrl_output = (outs Optional<AnyType>:$ctrl_output);
  let trait_attrs = (ins
    I32ElementsAttr:$operand_segment_sizes,
    I32ElementsAttr:$result_segment_sizes
  );
}

class OneFlow_UserBaseOp<string mnemonic, list<OpTrait> traits = [OneFlow_AlternativeOp]> :
        OneFlow_BaseOp<mnemonic, traits> {
    let summary = "";
    let user_op_attrs = (ins
      StrAttr:$op_type_name,
      // NOTE: vector types must have positive constant sizes, so we can't use I32ElementsAttr
      I32ArrayAttr:$input_sizes,
      I32ArrayAttr:$output_sizes
    );
}

// Why don't we merge ctrl in/out and data in/out into operand_segment/result_segment_sizes?
// 1. We only need to erase operand_segment/result_segment_sizes when we are creating a concrete user op
// 2. Isolating data and ctrl make debug easier and produced IR more human-readable
class OneFlow_UserBaseWithCtrlOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_UserBaseOp<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, AttrSizedResultSegments, DeclareOpInterfaceMethods<ControlEdgeCompatibleInterface>])> {
    let summary = "";
    let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
    let ctrl_output = (outs Optional<AnyType>:$ctrl_output);
    let trait_attrs = (ins
      I32ElementsAttr:$operand_segment_sizes,
      I32ElementsAttr:$result_segment_sizes
    );
}


class OneFlow_ConvolutionBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
    let summary = "OneFlow convolution operation";
    let description = [{
    "The convolution operator consumes an input tensor and a filter, and"
    "computes the output."
    }];
    let input = (ins
      AnyType:$in,
      AnyType:$weight,
      Optional<AnyType>:$bias,
      Optional<AnyType>:$bias_multiplier
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
    let trait_attrs = (ins
      I32ElementsAttr:$operand_segment_sizes
    );
}

class OneFlow_TFPoolBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
    let summary = "OneFlow pooling operation, align with TensorFlow";
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

class OneFlow_TFPoolGradBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow pooling grad operation, align with TensorFlow";
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


class OneFlow_MaxPoolBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow Max Pooling operation";
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y,
    AnyType:$indice
  );
  let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    SI32ArrayAttr:$dilation,
    DefaultValuedAttr<BoolAttr, "false">:$return_indices,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode
  );
}

class OneFlow_AvgPoolBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow Average Pooling operation";
  let input = (ins
    AnyType:$x
  );
  let output = (outs
    AnyType:$y
  );
  let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode,
    DefaultValuedAttr<BoolAttr, "false">:$count_include_pad,
    DefaultValuedAttr<SI64Attr, "0">:$divisor_override
  );
}

class OneFlow_MaxPoolGradBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow Max Pooling Grad operation";
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$indice,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    SI32ArrayAttr:$dilation,
    DefaultValuedAttr<BoolAttr, "false">:$return_indices,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode
  );
}

class OneFlow_AvgPoolGradBaseOp<string mnemonic, list<OpTrait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow Average Pooling Grad operation";
  let input = (ins
    AnyType:$x,
    AnyType:$y,
    AnyType:$dy
  );
  let output = (outs
    AnyType:$dx
  );
  let attrs = (ins
    SI32ArrayAttr:$padding,
    StrAttr:$data_format,
    SI32ArrayAttr:$kernel_size,
    SI32ArrayAttr:$stride,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode,
    DefaultValuedAttr<BoolAttr, "false">:$count_include_pad,
    DefaultValuedAttr<SI64Attr, "0">:$divisor_override
  );
}

class OneFlow_AdaptivePoolBaseOp<string mnemonic, list<OpTrait> traits = []> :
       OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow adaptive pool operation";
  let input = (ins
    AnyType:$x
  );
  let output = (outs AnyType:$y);
  let attrs = (ins
    SI64ArrayAttr:$output_size
  );
}

class OneFlow_AdaptivePoolGradBaseOp<string mnemonic, list<OpTrait> traits = []> :
       OneFlow_BaseOp<mnemonic, !listconcat(traits, [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>])> {
  let summary = "OneFlow adaptive pool operation";
  let input = (ins
    AnyType:$x,
    AnyType:$dy
  );
  let output = (outs AnyType:$dx);
  let attrs = (ins
    SI64ArrayAttr:$output_size
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

class OneFlow_JobOp<string mnemonic, list<OpTrait> traits = []> :
        Op<OneFlow_Dialect, mnemonic, !listconcat(traits, [FunctionLike, Symbol])> {
  let regions = (region AnyRegion:$body);

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    OptionalAttr<StrAttr>:$sym_visibility
  );

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type)
  >];

  let extraClassDeclaration = [{
    // FuncOp clone(BlockAndValueMapping &mapper);
    // FuncOp clone();

    // void cloneInto(FuncOp dest, BlockAndValueMapping &mapper);

    // ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    // ArrayRef<Type> getCallableResults() { return getType().getResults(); }

    // bool isDeclaration() { return isExternal(); }

  private:
    friend class OpTrait::FunctionLike<Job>;

    unsigned getNumFuncArguments() { return getType().getInputs().size(); }

    unsigned getNumFuncResults() { return getType().getResults().size(); }

    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];
}

#define GET_ONEFLOW_BASE_OP_DEFINITIONS
include "OneFlow/OneFlowUserOpGen.td"

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWBASE_H_
