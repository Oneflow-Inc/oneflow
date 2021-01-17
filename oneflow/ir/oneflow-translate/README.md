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
- MLIR exporters and and exporters should take care of blob names so other components don't touch it.

### About SBP signature
- There should be a sharding op to store SBP information.
- Reusing built-in tensor types is pratical and makes it easy to resuse pass interfaces.
- Implementing a tensor type with SBP is actually working agaist MLIR because pass in MLIR works better with operations.
### Dump generic user op to protobuf
1. find original user op to get bn, convert Variadic operands and outputs to `ArgDef`, keeping the same order
2. convert attributes

### Basic principles for a legit rewrite

1. Source op of control edge shouldn't be erased
2. Erasing, creating op shouldn't introduce boxing
3. Results' shapes should stay identical
### Information not included in OpConf

- There are information in job not included in `OpConf`:
```protobuf
message JobHelperConf {
  map<string, LogicalBlobIdPairs> tag2lbi_relations = 1;
  map<string, OpNameRelations> tag2op_name_relations = 2;
  map<string, OpTimeShape> op_name2op_time_shape = 3;
  map<string, BlobDescProto> lbn2logical_blob_desc = 4;
  map<string, int64> lbn2logical_object_id = 5;
  map<string, OptInt64> lbn2batch_axis = 6;
  optional OpBlobArgPairs identical_sbp_oba_pairs = 7;
  optional LbiDiffWatcherInfo lbi_diff_watcher_info = 8;
}

message JobParallelViewConf {
  map<string, SbpSignature> op_name2sbp_signature_conf = 1;
  map<string, bool> op_name2is_mirrored_parallel_view = 2;
}
```

- Create callbacks wrapping `JobBuilder` MLIR can call when it is erasing/building operations.

### Caveat
* Variadic operands and outputs might get erased
