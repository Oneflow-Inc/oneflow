#include "OneFlow/MLIROneFlowTranslation.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Translation.h"
#include "mlir/Support/LogicalResult.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::registerFromOneFlowJobTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
