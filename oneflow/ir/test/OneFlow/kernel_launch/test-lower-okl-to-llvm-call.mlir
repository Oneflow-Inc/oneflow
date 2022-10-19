module {
  llvm.func @okl_compute(%arg0: !llvm.ptr<i8>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<i8> to !okl.launcher_ctx
    %1 = "okl.fetch_run_ctx"(%0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %2 = "okl.fetch_run_ctx"(%0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %3 = "okl.fetch_kernel"(%0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    %4 = "okl.fetch_kernel"(%0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    "okl.launch"(%1, %3) : (!okl.run_ctx, !okl.kernel) -> ()
    "okl.launch"(%2, %4) : (!okl.run_ctx, !okl.kernel) -> ()
    llvm.return
  }
}
