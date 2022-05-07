# Lifetime and hierarchy of resource involved in IREE integration

## High-level API perspective

### Option #1: compile `nn.graph` to a callable function

```python
class Res50(flow.nn.Graph):
    pass
res50_nn_graph = ReluGraph()
compiled_res50 = oneflow.nn.graph.compile(res50_nn_graph, backend="iree")
compiled_res50(image_tensor)
```

- has least impact on existing `nn.graph` implementation
- It is possible to reuse the memory of input tensor of `nn.graph`
- IREE compiled function has its own memory pool.
- IREE compiled function reuse `nn.graph`'s CUDA stream.

### Option #2: alternative way to build a `nn.graph`

```python
class Res50(flow.nn.Graph):
    def __init__(self):
        super().__init__(backend="iree")
        ...

    def build(self, x):
        ...

res50_nn_graph = ReluGraph()
compiled_res50(image_tensor)
```

- this could introduce big change to `nn.graph`. We need to refactor `nn.graph` to become multi-backend, a default-backend, iree-backend, xrt-backend, etc.
- oneflow could allocation a big chunk of memory for IREE compiled function.
- it could have its own memory model and dedicated CUDA stream.

## Packaging and interaction

- the best scenario is that we don't need to compile IREE from source
- we will publish a oneflow-iree pip package containing python code only. This package has oneflow and IREE of compatible version as dependencies.
- All C++ code to support IREE are compiled and shipped with oneflow itself
- use [IREE's extension mechanism](https://google.github.io/iree/extensions/#2-extend-host-code-with-custom-modules) to interact with oneflow. Which means there are some C functions in oneflow available for IREE to call by reading the declaration in oneflow generated-IR.

## Resources

### IREE compiled function

- it should be a wrapper around IREE compiled function, which has input tensor, output tensor, memory buffer.
- the best scenario is that we could read input tensor without copy
- the best scenario is that we could write to output tensor without copy
- the wrapper also holds the memory buffer, so that IREE compiled function can use it safely

### oneflow tensor

- depends on the implementation maybe we need function to make a tensor's raw pointer accessible to IREE

### oneflow cuda stream

- TODO: can we make IREE to use oneflow's cuda stream?

### nn.graph

- TODO: should `nn.graph` own IREE context, function wrapper, memory buffer?

### IREE context

- if we can use oneflow to allocation memory for IREE, we can have one iree context for each function
- if IREE has its own memory management, it makes more sense to have a global context.

## Lifetime

1. create a nn.graph or a IREE compiled function
   - in Option #1, the compiled function's has its own lifetime, controlled by python garbage collector.
   - in Option #2, nn.graph own the compiled function
2. create IREE context
3. IREE compiled function read from oneflow tensor as argument and write in-place to oneflow tensor as result
4. depend on different implementation, `nn.graph` of a function wrapper is responsible for unload the context/module from IREE.

## Other remaining questions?

- How IREE's asynchronous execution works with oneflow? Are there CUDA stream involved?
