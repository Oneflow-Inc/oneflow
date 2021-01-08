# OneFlow Translate
## Import OneFlow Job to MLIR and dump a new Job
```
job -> module
sub graph -> function
```

### Pipeline
- Lower case: OneFlow, upper case: MLIR
- [something]: a step, could be rewrite or other kinds of optimizations
    ```
    user op -> GENERIC USER OP -> DEFINED OP -> [OPTIMIZATION] -> GENERIC USER OP -> user op
    ```

### Dump generic user op to protobuf
1. find original user op to get bn, convert Variadic operands and outputs to `ArgDef`, keeping the same order
2. convert attributes

### Caveat
* Variadic operands and outputs might get erased
