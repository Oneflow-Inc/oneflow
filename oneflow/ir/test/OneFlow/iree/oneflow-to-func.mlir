// oneflow-opt -lower-oneflow-to-func %s | FileCheck %s
module {
   oneflow.job private @TestLowerFunc() {
    oneflow.return 
  }
}
