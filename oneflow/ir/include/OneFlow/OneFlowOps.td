#ifndef ONEFLOW_OPS
#define ONEFLOW_OPS

include "OneFlow/OneFlowDialect.td"
include "OneFlow/OneFlowEnums.td"
include "OneFlow/OneFlowInterfaces.td"
include "OneFlow/OneFlowBase.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Pass/PassBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
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

#ifndef REMOVE_ONEFLOW_MLIR_ONLY_OP_DEFINITIONS

def OneFlow_UserOp : OneFlow_UserBaseWithCtrlOp<"user", [OneFlow_IsImportCompatible]> {
  let summary = "";
  let input = (ins Variadic<AnyType>:$data_input);
  let output = (outs Variadic<AnyType>:$data_output);
  let attrs = (ins
    StrArrayAttr:$output_lbns
  );
  let hasCanonicalizer = 1;
}

def OneFlow_ConfOp : OneFlow_BaseOp<"conf", [OneFlow_IsImportCompatible]> {
  let summary = "This op is mainly used by create its adaptor in importing/exporting";
}

def OneFlow_SystemOp : OneFlow_Op<"system", [OneFlow_IsImportCompatible]> {
  let summary = "";
  let input = (ins Variadic<AnyType>:$data_input);
  let output = (outs Variadic<AnyType>:$data_output);
  let attrs = (ins
    StrArrayAttr:$input_bns,
    StrArrayAttr:$output_lbns,
    I32Attr:$op_type_case
  );
  let hasCanonicalizer = 1;
}

def F32ElementsAttr : FloatElementsAttr<32>;

def OneFlow_FrozenVariableOp : OneFlow_IROp<"variable_ir", [ConstantLike, NoSideEffect]> {
  let summary = "Auxiliary variable op for constant folding, only exists in IR.";
  let arguments = (ins
    F32ElementsAttr:$value,
    StrAttr:$op_name,
    StrAttr:$device_tag,
    StrArrayAttr:$device_name, // TODO: change device_name to dict and parse the literal fmt like "0:0-0"
    OptionalAttr<I64Attr>:$scope_symbol_id,
    OptionalAttr<I64ArrayAttr>:$hierarchy,
    StrArrayAttr:$nd_sbp
  );
  let results = (outs
    AnyType:$output
  );
  let hasFolder = 1;
}

def OneFlow_LossMarkerOp : OneFlow_IROp<"loss_marker", []> {
  let summary = "mark a result as loss to prevent its defining op from optimized away";
  let arguments = (ins Variadic<AnyType>:$loss);
}

def OneFlow_Add2Op : OneFlow_BaseOp<"add_n2", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<AlternativeOpTypeNameInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {
  let summary = "";
  let input = (ins
    AnyType:$in0,
    AnyType:$in1
  );
  let output = (outs AnyType:$out);
}

// JIT ops

def OneFlow_MlirJitOp : OneFlow_BaseOp<"mlir_jit", [ CallOpInterface, DeclareOpInterfaceMethods<UserOpCompatibleInterface> ] > {
  let input = (ins Variadic<AnyType>:$data_input);
  let output = (outs Variadic<AnyType>:$data_output);
  let attrs = (ins
    FlatSymbolRefAttr:$callee,
    StrAttr:$mlir_assembly
  );
  let builders = [
    OpBuilder<(ins "func::FuncOp":$callee,
      "NamedAttrList":$attributes,
      CArg<"ValueRange", "{}">:$data_input), [{
      $_state.addOperands(data_input);
      $_state.addAttributes(attributes);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
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

class OneFlow_ConcreteSystemOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [OneFlow_IsOpConfCompatible,
        OneFlow_IsImportCompatible,
        DeclareOpInterfaceMethods<ControlEdgeCompatibleInterface>])> {
  let input = (ins);
  let output = (ins);
  let ctrl_input = (ins Variadic<AnyType>:$ctrl_inputs);
  let ctrl_output = (outs Optional<AnyType>:$ctrl_output);
  dag required_attrs = (ins StrArrayAttr:$output_lbns);
  dag custom_attrs = (ins);
  let attrs = !con(
    required_attrs,
    custom_attrs
  );
  let hasCanonicalizer = 1;
}

def OneFlow_VariableOp : OneFlow_ConcreteSystemOp<"variable", [OneFlow_TensorSource]> {
  let summary = "";
  let input = (ins);
  let output = (outs AnyType:$output);
  let custom_attrs = (ins
    ShapeAttr:$shape,
    OptionalAttr<OneFlow_DataType>:$data_type,
    DefaultValuedAttr<StrAttr, "\"weight\"">:$model_name,
    DefaultValuedAttr<F32Attr, "0.0f">:$l1_regularization,
    DefaultValuedAttr<F32Attr, "0.0f">:$l2_regularization,
    DefaultValuedAttr<BoolAttr, "true">:$trainable,
    OptionalAttr<F32Attr>:$float_initializer,
    OptionalAttr<SI64Attr>:$integer_initializer,
    StrArrayAttr:$nd_sbp
  );
}

def OneFlow_InputOp : OneFlow_ConcreteSystemOp<"input", [OneFlow_TensorSource]> {
  let summary = "";
  let input = (ins AnyType:$input);
  let output = (outs AnyType:$output);
  let custom_attrs = (ins
    OptionalAttr<ShapeAttr>:$shape,
    OptionalAttr<OneFlow_DataType>:$data_type,
    OptionalAttr<BoolAttr>:$is_dynamic,
    OptionalAttr<StrArrayAttr>:$nd_sbp,
    OptionalAttr<StrAttr>:$job_name
  );
  let builders = [
    OpBuilder<(ins
      "::oneflow::OperatorConf":$op_conf
    )>
  ];
}

def OneFlow_OutputOp : OneFlow_ConcreteSystemOp<"output", [OneFlow_TensorSource]> {
  let summary = "";
  let input = (ins AnyType:$input);
  let output = (outs AnyType:$output);
  let custom_attrs = (ins
    OptionalAttr<ShapeAttr>:$shape,
    OptionalAttr<OneFlow_DataType>:$data_type,
    OptionalAttr<BoolAttr>:$is_dynamic,
    OptionalAttr<StrArrayAttr>:$nd_sbp,
    OptionalAttr<StrAttr>:$job_name
  );
}

def OneFlow_Job : Op<OneFlow_Dialect, "job", [FunctionOpInterface, IsolatedFromAbove, Symbol]>  {
  let regions = (region AnyRegion:$body);

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility
  );

  let builders = [OpBuilder<(ins
    "StringRef":$sys_name, "FunctionType":$function_type)
  >];

  let extraClassDeclaration = [{
    bool isDeclaration() { return isExternal(); }

   FunctionType getFunctionType() { return function_type(); }

   TypeAttr getFunctionTypeAttr() { return function_typeAttr(); }

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def OneFlow_ReturnOp : Op<OneFlow_Dialect, "return", [NoSideEffect, HasParent<"Job">,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a Job.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the job function that contains
    the operation. For example:

    ```mlir
      job @foo() -> tensor<2xf64> {
        ...
        oneflow.return %0 : tensor<2xf64>
      }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, llvm::None); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def OneFlow_NormalizationInferenceOp : OneFlow_NormalizationBaseOp<"normalization_infer", [DeclareOpInterfaceMethods<AlternativeOpTypeNameInterface>]> {
  let output = (outs
    OneFlow_Tensor:$y
  );
}

#endif // REMOVE_ONEFLOW_MLIR_ONLY_OP_DEFINITIONS

#endif // ONEFLOW_OPS

#ifndef ONEFLOW_PASSES
#define ONEFLOW_PASSES

def LowerOneFlowToTosaPass : Pass<"lower-oneflow-to-tosa", "ModuleOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createLowerOneFlowToTosaPass()";
  let dependentDialects = ["tosa::TosaDialect", "memref::MemRefDialect", "mlir::func::FuncDialect"];
}

def MapSCFToGPUPass : Pass<"gpu-greedy-parallel-loop-mapping", "ModuleOp"> {
  let summary = "Greedily maps all parallel loops to gpu hardware ids";
  let constructor = "mlir::oneflow::createMapSCFToGPUPass()";
  let dependentDialects = ["scf::SCFDialect"];
}

def BufferHostRegisterPass : Pass<"buffer-host-register", "func::FuncOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createBufferHostRegisterPass()";
  let dependentDialects = ["gpu::GPUDialect"];
}

def GpuCopyArgPass : Pass<"gpu-copy-arg", "func::FuncOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createGpuCopyArgPass()";
  let dependentDialects = ["memref::MemRefDialect", "gpu::GPUDialect"];
}

def OutlineJitFunctionPass : Pass<"outline-jit-function", "ModuleOp"> {
  let summary = "move ops could be jitted to jit function";
  let constructor = "mlir::oneflow::createOutlineJitFunctionPass()";
}

def FuseIntoExistingOpPass : Pass<"fuse-into-existing-op", "ModuleOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createFuseIntoExistingOpPass()";
}

def AutoNhwcPass : Pass<"auto-nhwc", "ModuleOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createAutoNhwcPass()";
}

def PreConvertInferenceOpPass : Pass<"pre-convert-inference-op", "ModuleOp"> {
  let summary = "Convert variable op to variable ir op for constant folding.";
  let constructor = "mlir::oneflow::createPreConvertInferenceOpPass()";
}

def ConvertInferenceOpPass : Pass<"convert-inference-op", "ModuleOp"> {
  let summary = "Convert ops to their inference version and rewrite them with a more performant equivalent in inference workflow.";
  let constructor = "mlir::oneflow::createConvertInferenceOpPass()";
}

def PostConvertInferenceOpPass : Pass<"post-convert-inference-op", "ModuleOp"> {
  let summary = "Convert variable ir op to variable op after contant folding.";
  let constructor = "mlir::oneflow::createPostConvertInferenceOpPass()";
}

#endif // ONEFLOW_PASSES
