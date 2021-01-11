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

### About blob name
- Blob names are the legacy concepts from the time when it takes a prototxt file to define a neural network in OneFlow.
- Blob name is a leaky abstraction, pervasive in graphs, operators, kernels and many other components of OneFlow.
- In IR, we should never allow blob names to penetrate MLIR dialect.
- MLIR exporters and and exporters should take care of blob names so other components don't touch it.

### Dump generic user op to protobuf
1. find original user op to get bn, convert Variadic operands and outputs to `ArgDef`, keeping the same order
2. convert attributes

### Caveat
* Variadic operands and outputs might get erased
