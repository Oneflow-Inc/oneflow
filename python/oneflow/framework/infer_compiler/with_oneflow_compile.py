import os
import types
import torch
import oneflow as flow
from oneflow.utils.tensor import to_torch
from typing import Any
from functools import wraps
from itertools import chain
from .transform.manager import transform_mgr
from .transform.custom_transform import set_default_registry
from .transform.builtin_transform import torch2oflow
from .utils.oneflow_exec_mode import oneflow_exec_mode, oneflow_exec_mode_enabled
from .utils.args_tree_util import input_output_processor
from .utils.log_utils import logger
from .utils.cost_util import cost_cnt
from .utils.param_utils import parse_device, check_device


class DualModule(torch.nn.Module):
    def __init__(self, torch_module, oneflow_module):
        torch.nn.Module.__init__(self)
        self._torch_module = torch_module
        self._oneflow_module = oneflow_module

    @property
    def oneflow_module(self):
        if self._oneflow_module is not None:
            return self._oneflow_module

        logger.debug(f"Convert {type(self._torch_module)} ...")
        self._oneflow_module = torch2oflow(self._torch_module)
        logger.debug(f"Convert {id(self._torch_module)=} done!")
        return self._oneflow_module

    @oneflow_module.deleter
    def oneflow_module(self):
        if self._oneflow_module:
            del self._oneflow_module
            setattr(self, "_oneflow_module", None)

    def to(self, *args, **kwargs):
        if oneflow_exec_mode_enabled():
            self._oneflow_module.to(*args, **kwargs)
        else:
            if self._oneflow_module is not None:
                args = [torch2oflow(v) for v in args]
                kwargs = {k: torch2oflow(v) for k, v in kwargs.items()}
                self._oneflow_module.to(*args, **kwargs)
                self._torch_module_to_with_check(*args, **kwargs)
            else:
                self._torch_module.to(*args, **kwargs)

    def _torch_module_to_with_check(self, *args, **kwargs):
        def _align_tensor(torch_module, oneflow_module):
            oneflow_tensor_list = set(
                [x for x, _ in oneflow_module.named_parameters()]
                + [x for x, _ in oneflow_module.named_buffers()]
            )
            for name, tensor in chain.from_iterable(
                [torch_module.named_parameters(), torch_module.named_buffers(),]
            ):
                if name not in oneflow_tensor_list:
                    tensor.data = tensor.to(*args, **kwargs)
                else:
                    oneflow_tensor = oneflow_module.get_parameter(name)
                    if oneflow_tensor is None:
                        tensor.data = tensor.to(*args, **kwargs)
                    elif tensor.data_ptr() != oneflow_tensor.data_ptr():
                        tensor.data = to_torch(oneflow_tensor.data)

        oneflow_module_list = set([x for x, _ in self._oneflow_module.named_modules()])
        for name, module in self._torch_module.named_modules():
            if name not in oneflow_module_list:
                module.to(*args, **kwargs)
            else:
                _align_tensor(module, self._oneflow_module.get_submodule(name))

    def __getattr__(self, name):
        if name == "_torch_module":
            return self._modules[name]
        if name == "_oneflow_module":
            return super().__getattribute__(name)

        torch_attr = getattr(self._torch_module, name)
        oneflow_attr = (
            None
            if self._oneflow_module is None
            else getattr(self._oneflow_module, name)
        )
        if isinstance(torch_attr, torch.nn.ModuleList):
            oneflow_attr = (
                [None] * len(torch_attr) if oneflow_attr is None else oneflow_attr
            )
            return DualModuleList(torch_attr, oneflow_attr)

        elif isinstance(torch_attr, torch.nn.Module):
            return get_mixed_dual_module(torch_attr.__class__)(torch_attr, oneflow_attr)
        else:
            return oneflow_attr if oneflow_exec_mode_enabled() else torch_attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["_torch_module", "_oneflow_module"]:
            super().__setattr__(name, value)
        else:  # TODO: aviod memory up when set attr
            if self._oneflow_module is not None:
                v = torch2oflow(value)
                if isinstance(v, flow.Tensor):
                    obj = getattr(self._oneflow_module, name)
                    obj.copy_(v)
                else:
                    setattr(self._oneflow_module, name, v)
            setattr(self._torch_module, name, value)


class DualModuleList(torch.nn.ModuleList):
    def __init__(self, torch_modules, oneflow_modules):
        super().__init__()
        assert len(torch_modules) == len(oneflow_modules)
        self._torch_modules = torch_modules
        self._oneflow_modules = oneflow_modules
        dual_modules = []
        for torch_module, oneflow_module in zip(
            self._torch_modules, self._oneflow_modules
        ):
            dual_modules.append(
                get_mixed_dual_module(torch_module.__class__)(
                    torch_module, oneflow_module
                )
            )
        # clear self._modules since `self._torch_modules = torch_modules` will append a module to self._modules
        self._modules.clear()
        self += dual_modules

    def __setitem__(self, idx: int, module: DualModule):
        idx = self._get_abs_string_index(idx)
        setattr(self._torch_modules, str(idx), module._torch_module)
        setattr(self._oneflow_modules, str(idx), module._oneflow_module)
        return setattr(self, str(idx), module)

    def __setattr__(self, key, value):
        if key in ("_torch_modules", "_oneflow_modules"):
            return object.__setattr__(self, key, value)
        if isinstance(value, DualModule):
            setattr(self._torch_modules, key, value._torch_module)
            setattr(self._oneflow_modules, key, value._oneflow_module)
        else:
            setattr(self._torch_modules, key, value)
            value = torch2oflow(value)
            setattr(self._oneflow_modules, key, value)
        return object.__setattr__(self, key, value)


def get_mixed_dual_module(module_cls):
    class MixedDualModule(DualModule, module_cls):
        def __init__(self, torch_module, oneflow_module):
            DualModule.__init__(self, torch_module, oneflow_module)

    return MixedDualModule


def graph_file_management(func):
    @wraps(func)
    def wrapper(self: "DeployableModule", *args, **kwargs):
        graph_file = self._deployable_module_options.get("graph_file", None)

        # Load graph file
        if graph_file is not None:
            try:
                if not os.path.exists(graph_file):
                    logger.warning(
                        f"Graph file {graph_file} not exists!, will generate graph."
                    )

                else:
                    graph_device = self._deployable_module_options.get(
                        "graph_file_device", None
                    )

                    self.load_graph(graph_file, torch2oflow(graph_device))
                    logger.info(f"Load graph file: {graph_file}")

                    graph_file = None
                    self._deployable_module_options["graph_file"] = None

            except Exception as e:
                logger.error(f"Load graph file: {graph_file} failed! {e=}")

        ret = func(self, *args, **kwargs)

        # Save graph file
        if graph_file is not None:
            try:
                if graph_file is not None:
                    os.makedirs(os.path.dirname(graph_file), exist_ok=True)
                    self.save_graph(graph_file)
                    logger.info(f"Save graph file: {graph_file} done!")
            except Exception as e:
                logger.error(f"Save graph file: {graph_file} failed! {e=}")
            finally:
                self._deployable_module_options["graph_file"] = None

        return ret

    return wrapper


def handle_deployable_exception(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if transform_mgr.debug_mode:
            return func(self, *args, **kwargs)
        else:
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e=}")
                logger.warning("Recompile oneflow module ...")
                del self._deployable_module_model.oneflow_module
                self._deployable_module_dpl_graph = None
                return func(self, *args, **kwargs)

    return wrapper


class DeployableModule(torch.nn.Module):
    def __init__(
        self,
        torch_module,
        oneflow_module,
        use_graph=True,
        options={},
        graph_path=None,
        graph_device=None,
    ):
        torch.nn.Module.__init__(self)
        self._deployable_module_model = get_mixed_dual_module(torch_module.__class__)(
            torch_module, oneflow_module
        )
        self._deployable_module_use_graph = use_graph
        self._deployable_module_options = options
        self._deployable_module_dpl_graph = None
        self._is_raw_deployable_module = True

    @classmethod
    def from_existing(cls, existing_module, use_graph=None, options=None):
        torch_module = existing_module._deployable_module_model._torch_module
        oneflow_module = existing_module._deployable_module_model._oneflow_module
        instance = cls(torch_module, oneflow_module, use_graph, options)
        instance._deployable_module_dpl_graph = (
            existing_module._deployable_module_dpl_graph if use_graph else None
        )
        return instance

    def get_graph(self):
        if self._deployable_module_dpl_graph is not None:
            return self._deployable_module_dpl_graph
        if "size" in self._deployable_module_options:
            size = self._deployable_module_options["size"]
        else:
            size = 9
        if "all_dynamic" in self._deployable_module_options:
            all_dynamic = self._deployable_module_options["all_dynamic"]
        else:
            all_dynamic = False
        self._deployable_module_dpl_graph = get_oneflow_graph(
            self._deployable_module_model.oneflow_module, size, all_dynamic
        )
        if "debug" in self._deployable_module_options:
            self._deployable_module_dpl_graph.debug(
                self._deployable_module_options["debug"]
            )
        return self._deployable_module_dpl_graph

    @input_output_processor
    @handle_deployable_exception
    @graph_file_management
    def apply_model(self, *args, **kwargs):
        if self._deployable_module_use_graph:
            dpl_graph = self.get_graph()
            with oneflow_exec_mode():
                output = dpl_graph(*args, **kwargs)
        else:
            with oneflow_exec_mode():
                output = self._deployable_module_model.oneflow_module.apply_model(
                    *args, **kwargs
                )
        return output

    @input_output_processor
    @handle_deployable_exception
    @graph_file_management
    def __call__(self, *args, **kwargs):
        if self._deployable_module_use_graph:
            dpl_graph = self.get_graph()
            with oneflow_exec_mode():
                output = dpl_graph(*args, **kwargs)
        else:
            with oneflow_exec_mode():
                output = self._deployable_module_model.oneflow_module(*args, **kwargs)
        return output

    def to(self, *args, **kwargs):
        if self._deployable_module_dpl_graph is None:
            self._deployable_module_model.to(*args, **kwargs)
            return self

        # assert the target device is same as graph device
        target_device = parse_device(args, kwargs)
        if (
            target_device is not None
            and len(self._deployable_module_dpl_graph._blocks) > 0
        ):
            current_device = next(self._deployable_module_dpl_graph._state()).device
            if not check_device(current_device, target_device):
                raise RuntimeError(
                    f"After graph built, the device of graph can't be modified, current device: {current_device}, target device: {target_device}"
                )
        self._deployable_module_model.to(*args, **kwargs)
        return self

    # TODO(): Just for transformers VAE decoder
    @input_output_processor
    @handle_deployable_exception
    @graph_file_management
    def decode(self, *args, **kwargs):
        if self._deployable_module_use_graph:

            def _build(graph, *args, **kwargs):
                return graph.model.decode(*args, **kwargs)

            dpl_graph = self.get_graph()
            dpl_graph.build = types.MethodType(_build, dpl_graph)
            with oneflow_exec_mode():
                output = dpl_graph(*args, **kwargs)
        else:
            with oneflow_exec_mode():
                output = self._deployable_module_model.oneflow_module.decode(
                    *args, **kwargs
                )
        return output

    def __getattr__(self, name):
        if name in self._modules:
            return self._modules[name]
        return getattr(self._deployable_module_model, name)

    def load_graph(self, file_path, device=None, run_warmup=True):
        self.get_graph().warmup_with_load(file_path, device, run_warmup)

    def warmup_with_load(self, file_path, device=None, run_warmup=True):
        self.get_graph().warmup_with_load(file_path, device, run_warmup)

    def save_graph(self, file_path):
        self.get_graph().save_graph(file_path)


class OneflowGraph(flow.nn.Graph):
    @flow.nn.Graph.with_dynamic_input_shape()
    def __init__(self, model):
        super().__init__(enable_get_runtime_state_dict=True)
        self.model = model
        self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.config.allow_fuse_add_to_output(True)

        os.environ["ONEFLOW_GRAPH_DELAY_VARIABLE_OP_EXECUTION"] = "1"
        os.environ["ONEFLOW_MLIR_CSE"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_OPS_WITH_BACKWARD_IMPL"] = "1"
        os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
        os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_CONV_BIAS"] = "1"
        os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
        os.environ["ONEFLOW_KERNEL_GEMM_CUTLASS_IMPL_ENABLE_TUNING_WARMUP"] = "1"
        os.environ["ONEFLOW_KERNEL_CONV_ENABLE_CUTLASS_IMPL"] = "1"
        os.environ["ONEFLOW_KERNEL_GEMM_ENABLE_CUTLASS_IMPL"] = "1"
        os.environ["ONEFLOW_CONV_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION"] = "1"
        os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"
        # os.environ["ONEFLOW_KERNEL_GLU_ENABLE_DUAL_GEMM_IMPL"] = "0"
        os.environ["ONEFLOW_MLIR_GROUP_MATMUL_QUANT"] = "1"
        # TODO: enable this will cause the failure of multi resolution warmup
        # os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
        # os.environ["ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH"] = "1"

    def build(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @cost_cnt(transform_mgr.debug_mode)
    def warmup_with_load(self, file_path, device=None, run_warmup=True):
        state_dict = flow.load(file_path)
        if device is not None:
            state_dict = flow.nn.Graph.runtime_state_dict_to(state_dict, device)
        self.load_runtime_state_dict(state_dict, warmup_with_run=run_warmup)

    @cost_cnt(transform_mgr.debug_mode)
    def save_graph(self, file_path):
        state_dict = self.runtime_state_dict()
        flow.save(state_dict, file_path)


def get_oneflow_graph(model, size=9, all_dynamic=False):
    g = OneflowGraph(model)
    g._dynamic_input_graph_cache.set_cache_size(size)
    g._dynamic_input_graph_cache.enable_shared(not all_dynamic)
    return g


def state_dict_hook(module, state_dict, prefix, local_metadata):
    pytorch_key_prefix = "_deployable_module_model._torch_module."
    new_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        # _deployable_module_model._torch_module.out.2.weight => out.2.weight
        if k.startswith(pytorch_key_prefix):
            new_k = k[len(pytorch_key_prefix) :]
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# Return a DeployableModule that using module_cls as it's parent class.
def get_mixed_deployable_module(module_cls):
    class MixedDeployableModule(DeployableModule, module_cls):
        def __init__(
            self,
            torch_module,
            oneflow_module,
            use_graph=True,
            options={},
            graph_path=None,
            graph_device=None,
        ):
            DeployableModule.__init__(
                self,
                torch_module,
                oneflow_module,
                use_graph,
                options,
                graph_path,
                graph_device,
            )
            self._is_raw_deployable_module = False

        @classmethod
        def from_existing(cls, existing_module, use_graph=None, options=None):
            torch_module = existing_module._deployable_module_model._torch_module
            oneflow_module = existing_module._deployable_module_model._oneflow_module
            instance = cls(torch_module, oneflow_module, use_graph, options)
            instance._deployable_module_dpl_graph = (
                existing_module._deployable_module_dpl_graph if use_graph else None
            )
            return instance

    return MixedDeployableModule


def oneflow_compile(
    torch_module: torch.nn.Module,
    *,
    use_graph=True,
    options={},
):
    set_default_registry()

    def wrap_module(module):
        if isinstance(module, DeployableModule):
            assert not module._is_raw_deployable_module
            return module.__class__.from_existing(module, use_graph, options)
        else:
            return get_mixed_deployable_module(module.__class__)(
                module, None, use_graph, options
            )

    model = wrap_module(torch_module)
    assert isinstance(model, DeployableModule)
    assert isinstance(model, torch_module.__class__)
    model._register_state_dict_hook(state_dict_hook)

    return model
