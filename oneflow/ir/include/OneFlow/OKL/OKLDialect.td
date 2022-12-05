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
#ifndef ONEFLOW_IR_INCLUDE_OKL_OKLDIALECT
#define ONEFLOW_IR_INCLUDE_OKL_OKLDIALECT

include "mlir/IR/OpBase.td"

def OKL_Dialect : Dialect {
    let name = "okl";
    let summary = "OneFlow Kernel Launch Dialect.";
    let description = [{
        This dialect is the IR of abstract represent of OneFlow Kernel Launch Op.
    }];
    let cppNamespace = "::mlir::okl";
    let dependentDialects = [
        "func::FuncDialect"
    ];
    let useDefaultTypePrinterParser = 1;
}

#endif // ONEFLOW_IR_INCLUDE_OKL_OKLDIALECT
