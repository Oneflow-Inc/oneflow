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
    user op ->  OPAQUE USER OP -> CONCRETE OP -> [OPTIMIZATION] -> user op
    system op ->  OPAQUE SYSTEM OP -> system op
    ```

### About blob name
- MLIR exporters and and exporters should take care of blob names so other components don't touch it.

### About SBP signature
- There should be a sharding op to store SBP information.
- Reusing built-in tensor types is pratical and makes it easy to resuse pass interfaces.
- Implementing a tensor type with SBP is actually working agaist MLIR because pass in MLIR works better with operations.

### Basic principles for a legit rewrite

1. Source op of control edge shouldn't be erased
2. Erasing, creating op shouldn't introduce boxing
3. Results' shapes should stay identical
### Information not included in OpConf

- There are information in job not included in `OpConf`:
```protobuf
message JobHelperConf {
  map<string, LogicalBlobIdPairs> tag2lbi_relations = 1;
  ...
}

message JobParallelViewConf {
  ...
}
```

- Create callbacks wrapping `JobBuilder` MLIR can call to update job helperconfs when it is erasing/building operations.
