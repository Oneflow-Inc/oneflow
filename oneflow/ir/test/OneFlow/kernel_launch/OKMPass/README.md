## To test inpace reshape compiling
```bash
FILECHECK_OPTS="-vv --dump-input=always --color"  LIT_OPTS="-a --filter=inplace_reshape.mlir -j1" cmake --build build -t c1
```
