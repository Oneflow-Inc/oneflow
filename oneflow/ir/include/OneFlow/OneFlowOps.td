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

def OneFlow_SystemOp : OneFlow_Op<"system", [OneFlow_IsImportCompatible]> {
  let summary = "";
  let input = (ins Variadic<AnyType>:$data_input);
  let output = (outs Variadic<AnyType>:$data_output);
  let attrs = (ins
    StrArrayAttr:$input_bns,
    StrArrayAttr:$output_lbns
  );
  let hasCanonicalizer = 1;
}

def OneFlow_Add2Op : OneFlow_BaseOp<"add_n2", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<AlternativeOpTypeNameInterface>]> {
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
    TypeAttr:$type,
    OptionalAttr<StrAttr>:$sym_visibility
  );

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type)
  >];

  let extraClassDeclaration = [{
    bool isDeclaration() { return isExternal(); }

    /// Returns the type of this function.
    /// TODO: use the ODS `type`
    FunctionType getType() {
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getType().getResults(); }

    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];

  let parser = [{ return ::mlir::oneflow::parseJob(parser, result); }];
  let printer = [{ return ::mlir::oneflow::print(*this, p); }];
  let verifier = [{ return ::mlir::oneflow::verify(*this); }];
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

  let printer = [{ return ::mlir::oneflow::print(p, *this); }];
  let verifier = [{ return ::mlir::oneflow::verify(*this); }];
  let parser = [{ return ::mlir::oneflow::parse$cppClass(parser, result); }];
}

#endif // REMOVE_ONEFLOW_MLIR_ONLY_OP_DEFINITIONS

#endif // ONEFLOW_OPS

#ifndef ONEFLOW_PASSES
#define ONEFLOW_PASSES

def LowerOneFlowToTosaPass : Pass<"lower-oneflow-to-tosa", "ModuleOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createLowerOneFlowToTosaPass()";
  let dependentDialects = ["tosa::TosaDialect", "memref::MemRefDialect", "StandardOpsDialect"];
}

def MapSCFToGPUPass : Pass<"gpu-greedy-parallel-loop-mapping", "ModuleOp"> {
  let summary = "Greedily maps all parallel loops to gpu hardware ids";
  let constructor = "mlir::oneflow::createMapSCFToGPUPass()";
  let dependentDialects = ["scf::SCFDialect"];
}

def BufferHostRegisterPass : Pass<"buffer-host-register", "FuncOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createBufferHostRegisterPass()";
  let dependentDialects = ["gpu::GPUDialect"];
}

def GpuCopyArgPass : Pass<"gpu-copy-arg", "FuncOp"> {
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

#endif // ONEFLOW_PASSES
