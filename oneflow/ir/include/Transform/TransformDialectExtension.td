#ifndef ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_TD_
#define ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_TD_

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Bufferization/IR/BufferizationEnums.td"

def ApplyPatternsOp : Op<Transform_Dialect, "oneflow.apply_patterns",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{
    Modified from iree project: https://github.com/openxla/iree
    Greedily applies patterns as specified by its attributes.

    Must be applied to an op with trait IsolatedFromAbove since the
    GreedyPatternRewriter asserts those. Internally, uses the tracking rewriter
    to preserve handles to payload operations nested within operations
    associated with `target`. Fails if tracking cannot find replacement for a
    payload operation. This may become controllable with an attribute in the
    future.

    Returns the IsolatedFromAbove op whose content it has modified for better
    chaining APIs.

    #### Return modes:

    This operation applies a set of patterns specified by attributes. To apply
    these patterns, this operation must target an operation that is isolated
    from above, otherwise the transform definitely fails.

    If the pattern application fails, or if the underlying listener fails to
    capture op handles, the transformation definitely fails.

    Otherwise the transformation is successful.

    This operation does not consume the target handle and does not produce any
    handle.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       UnitAttr:$cse,
                       UnitAttr:$memref_canonicalization,
                       UnitAttr:$canonicalization);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` functional-type($target, results)";
  let cppNamespace = "mlir::oneflow::transform_dialect";

  let builders = [
    OpBuilder<(ins "Value":$target,
                   "const ApplyPatternsOpPatterns &":$patterns)>
  ];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def OneShotBufferizeOp
    : Op<Transform_Dialect, "oneflow.one_shot_bufferize",
        [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
         DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Modified from llvm project.

    Indicates that the given `target` op should be bufferized with One-Shot
    Bufferize. The bufferization can be configured with various attributes that
    corresponding to options in `BufferizationOptions` and the
    `one-shot-bufferize` pass. More information can be found in the pass
    documentation.

    The targeted ops must be modules or functions. This is because there is
    always a single, bufferized replacement op for such targets.

    Note: Only ops that implement `BufferizableOpInterface` are bufferized. All
    other ops are ignored if `allow_unknown_ops`. If `allow_unknown_ops` is
    unset, this transform fails when an unknown/non-bufferizable op is found.
    Many ops implement `BufferizableOpInterface` via an external model. These
    external models must be registered when applying this transform op;
    otherwise, said ops would be considered non-bufferizable.

    #### Return modes

    This operation consumes the `target` handle and produces the `transformed`
    handle.
  }];

  let arguments = (
      ins TransformHandleTypeInterface:$target,
      OptionalAttr<LayoutMapOption>:$function_boundary_type_conversion,
      DefaultValuedAttr<BoolAttr, "false">:$support_gpu,
      DefaultValuedAttr<BoolAttr, "false">:$allow_return_allocs,
      DefaultValuedAttr<BoolAttr, "false">:$allow_unknown_ops,
      DefaultValuedAttr<BoolAttr, "false">:$bufferize_function_boundaries,
      DefaultValuedAttr<BoolAttr, "true">:$create_deallocs,
      DefaultValuedAttr<BoolAttr, "false">:$test_analysis_only,
      DefaultValuedAttr<BoolAttr, "false">:$print_conflicts);

  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = [{
    (`layout` `{` $function_boundary_type_conversion^ `}`)?
    $target attr-dict `:` functional-type($target, results)
  }];

  let cppNamespace = "mlir::oneflow::transform_dialect";
}

#endif // ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_TD_
