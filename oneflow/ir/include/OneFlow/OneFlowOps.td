#ifndef ONEFLOW_OPS
#define ONEFLOW_OPS

include "OneFlow/OneFlowDialect.td"
include "OneFlow/OneFlowEnums.td"
include "OneFlow/OneFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Pass/PassBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "OneFlow/OneFlowBase.td"

def OneFlow_UserOp : OneFlow_UserBaseWithCtrlOp<"user", [OneFlow_IsImportCompatible]> {
  let summary = "";
  let attrs = (ins
    StrArrayAttr:$output_lbns
  );
  let hasCanonicalizer = 1;
}

def OneFlow_SystemOp : OneFlow_Op<"system", [OneFlow_IsImportCompatible]> {
  let summary = "";
  let attrs = (ins
    StrArrayAttr:$input_bns,
    StrArrayAttr:$output_lbns
  );
  let hasCanonicalizer = 1;
}

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
