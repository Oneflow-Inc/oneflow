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

- ### OneFlow Kenerl Memory (OKM) Dialect
In the `include` and `lib` directories, there are definitions of MLIR OKM dialect and its operators.
OKM is a dialect which support oneflow using mlir memref style and use-def flow  to optimize memory usage.

- ### OneFlow Kernel Launch (OKL) dialect
In the `include` and `lib` directories, there are definitions of MLIR OKL dialect and its operators.
OKL is a dialect which support oneflow kernel ops launched as a a llvm dialect callee.
## Parallel Signature

- There is parallel signature as 0 for OneFlow Ops in MLIR. It is implemented as MLIR dialect attribute. Some examples:
    - 1D SBP
        ```mlir
        %100 = "oneflow.relu"(%99) {parallel = #sbp.parallel<[#sbp.S<0>] -> [#sbp.S<0>]>, ...
        ```
    - multiple inputs and outputs 1D SBP
        ```mlir
        %102 = "oneflow.add_n2"(%101, %97) {parallel = #sbp.parallel<[#sbp.S<0>, #sbp.S<0>] -> [#sbp.S<0>]>, ...
        ```
    - 2D SBP `matmul`
        ```
        %120 = "oneflow.matmul"(%119, %output_105) {parallel = #sbp.parallel<[[#sbp.S<0>, #sbp.P], #sbp.S<0>] -> [#sbp.S<0>]>, ...
        ```

- To avoid confusion and potential parsing error, use the term "parallel" instead of using "sbp" but conceptually and documentally there are the same.

### Principle
- In IR, The signature should be orthogonal to device placement information althogh in some passes they might be related to each other.

## Development

- To run all the regression tests. The `-j3` option for [`LIT`](https://llvm.org/docs/CommandGuide/lit.html) is to prevent OOM on GPU.
    ```bash
    LIT_OPTS="-j3" cmake --build build -t c1 -j24
    ```
