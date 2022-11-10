namespace {

struct GeneratedPDLLPattern0 : ::mlir::PDLPatternModule {
  GeneratedPDLLPattern0(::mlir::MLIRContext *context)
    : ::mlir::PDLPatternModule(::mlir::parseSourceString<::mlir::ModuleOp>(
R"mlir(pdl.pattern : benefit(0) {
  %0 = operands loc("expr.pdll":1:18)
  %1 = types loc("expr.pdll":1:18)
  %2 = operation(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>) loc("expr.pdll":1:18)
  rewrite %2 {
    erase %2 loc("expr.pdll":1:12)
  } loc("expr.pdll":1:12)
} loc("expr.pdll":1:1)
    )mlir", context)) {
  }
};

} // end namespace

static void LLVM_ATTRIBUTE_UNUSED populateGeneratedPDLLPatterns(::mlir::RewritePatternSet &patterns) {
  patterns.add<GeneratedPDLLPattern0>(patterns.getContext());
}
