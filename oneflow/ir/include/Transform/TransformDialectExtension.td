#ifndef ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_TD_
#define ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_TD_

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"

class ProduceNoneProto<string mnemonic, list<Trait> traits = []> : 
  Op<Transform_Dialect, mnemonic,
    traits # [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` functional-type($target, results)";
  let cppNamespace = "mlir::oneflow::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);

    void getEffects(
        SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
      ::mlir::transform::onlyReadsHandle(getTarget(), effects);
      ::mlir::transform::modifiesPayload(effects);
    }
  }];
}

def CSEOp : ProduceNoneProto<"oneflow.cse"> {
  let description = [{
    cse in transform dialect.
  }];
}

def CanonicalizationOp : ProduceNoneProto<"oneflow.canonicalization"> {
  let description = [{
    canonicalization in transform dialect.
  }];
}

def ExplicitLinalgOutcomeOp : ProduceNoneProto<"oneflow.explicit_linalg_outcome"> {
  let description = [{
    fold unit-extent dimensions in operands/results of linalg ops on tensors via rank-reducing slice in transform dialect.
  }];
}

def EliminateCopyOp : ProduceNoneProto<"oneflow.eliminate_copy"> {
  let description = [{
    eliminate memref.copy if its target equals to source or comes from block arguments.
  }];
}

def FoldAllocOp : ProduceNoneProto<"oneflow.fold_alloc"> {
  let description = [{
    fold memref.alloc to a single one and subview on it.
  }];
}

def ResultsToOutParamsOp : ProduceNoneProto<"oneflow.results_to_out_params"> {
  let description = [{
    move results to out params.
  }];
}

#endif // ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_TD_
