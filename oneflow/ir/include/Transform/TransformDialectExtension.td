#ifndef ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_TD_
#define ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_TD_

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"

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

    The following additive attributes can be set, they add patterns in an
    unspecified order:
      - canonicalization: adds all the canonicalization patterns of all
      registered dialects and ops.


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

#endif // ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_TD_
