# OneFlow IR

OneFlow IR, a MLIR dialect

## Code style

Inevitably, developers maintaining OneFlow IR would face these challenges:
- Debugging components related to IR, compiler could be complicated and peculiar.
- IR subsystems should follow latest changes of OneFlow and MLIR closely.

To address these problems,
within the IR source code directory,
there are some rules must be enforced for all the optimizers, importers, exporters, runners:
- separate library, include, target
- MLIR-releted code should follow the style and paradigm of MLIR and LLVM closely
- ensure every component could be independently compiled and tested
    - there should be one `CMakeLists.txt` in every sub-directory
    - don't link anything from OneFlow unless it is necessary for the feature

## Major components
- ### oneflow-translate
Everything related to MLIR-OneFlow translation. [read more](oneflow-translate/README.md)

- ### oneflow-opt
Optimizations on OneFlow MLIR dialect. A CLI to optimize .mlir file. [read more](oneflow-opt/README.md)

- ### OneFlow dialect
In the `include` and `lib` directories, there are definitions of MLIR OneFlow dialect and its operators.

## Development

- To run all the regression tests. The `-j3` option for [`LIT`](https://llvm.org/docs/CommandGuide/lit.html) is to prevent OOM on GPU.
    ```bash
    LIT_OPTS="-j3" cmake --build build -t c1 -j24
    ```
