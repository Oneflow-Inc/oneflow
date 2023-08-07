/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "OneFlow/SBP/SBPDialect.h"
#include "OneFlow/OKL/OKLDialect.h"
#include "OneFlow/OKM/OKMDialect.h"
#include "OneFlow/OKL/passes.h"
#include "OneFlow/OKM/passes.h"
#include "Transform/TransformDialectExtension.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

const auto global_cse_state = std::make_shared<mlir::oneflow::CSEState>();

int32_t main(int32_t argc, char** argv) {
  ::oneflow::Singleton<::oneflow::ProcessCtx>::New();
  mlir::registerAllPasses();
  mlir::oneflow::registerCSEPasses(global_cse_state);
  mlir::oneflow::registerPasses();
  mlir::okm::registerPasses();
  mlir::okl::registerPasses();
  mlir::oneflow::transform_dialect::registerTransformDialectEraseSchedulePass();
  mlir::oneflow::transform_dialect::registerTransformDialectInterpreterPass();

  mlir::DialectRegistry registry;
  // Note: register all mlir dialect and their extension.
  mlir::registerAllDialects(registry);
  mlir::oneflow::transform_dialect::registerTransformDialectExtension(registry);
  registry.insert<mlir::okl::OKLDialect>();
  registry.insert<mlir::okm::OKMDialect>();
  registry.insert<mlir::sbp::SBPDialect>();
  registry.insert<mlir::oneflow::OneFlowDialect>();
  return failed(mlir::MlirOptMain(argc, argv, "OneFlow optimizer driver\n", registry));
}
