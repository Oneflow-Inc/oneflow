mlir生成的llvm最终的参数列表为：

 - 缓存池相关信息
 - 输入1相关信息 ... 输入n相关信息
 - 输出1相关信息 ... 输出n相关信息
 - stream 相关信息

基于上述abi设计相关pass
 - append-ofstream
 - insert-ofmempool