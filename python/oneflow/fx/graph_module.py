"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
import oneflow.nn as nn
import linecache
from typing import Type, Dict, List, Any, Union, Optional, Set
from .graph import Graph, _is_from_flow, _custom_builtins, PythonCode
import copy
import itertools
import sys
import traceback
from pathlib import Path
import os
import warnings

# normal exec loses the source code, however we can patch
# the linecache module to still recover it.
# using exec_with_source will add it to our local cache
_next_id = 0


def exec_with_source(src: str, globals: Dict[str, Any]):
    global _next_id
    key = f"<eval_with_key_{_next_id}>"
    _next_id += 1
    _eval_cache[key] = [line + "\n" for line in src.splitlines()]
    exec(compile(src, key, "exec"), globals)


# patch linecache so that any code we exec using exec_with_source
# works with inspect
_eval_cache: Dict[str, List[str]] = {}
_orig_getlines = linecache.getlines


def patched_getline(*args, **kwargs):
    if args[0] in _eval_cache:
        return _eval_cache[args[0]]
    return _orig_getlines(*args, **kwargs)


linecache.getlines = patched_getline


def _forward_from_src(src: str, globals: Dict[str, Any]):
    # avoid mutating the passed in dict
    globals_copy = globals.copy()
    exec_with_source(src, globals_copy)
    forward_fn = globals_copy["forward"]
    del globals_copy["forward"]
    return forward_fn


# copy an attribute value with qualified name 'target' from 'from_module' to 'to_module'
# This installs empty Modules where none exist yet if they are subpaths of target
def _copy_attr(from_module: flow.nn.Module, to_module: flow.nn.Module, target: str):
    *prefix, field = target.split(".")
    for item in prefix:
        f = getattr(from_module, item)
        t = getattr(to_module, item, None)
        if f is t:
            # we have already installed one of its parents
            # (e.g. target = root.linear.weight, but we have already installed root.linear)
            # once we install a parent, we no longer need to copy the children
            # since all the needed properties will already be present
            return

        if t is None:
            t = flow.nn.Module()
            setattr(to_module, item, t)
        from_module, to_module = f, t

    orig = getattr(from_module, field)
    # If it is a tensor and not a parameter attribute of a module, it should be a named buffer.
    # So, we register it as a named buffer in the target module.
    if isinstance(orig, flow.Tensor) and not isinstance(orig, flow.nn.Parameter):
        to_module.register_buffer(field, orig)
    else:
        setattr(to_module, field, orig)


# Assign attribute 'from_obj' to the qualified name 'target' on 'to_module
# This installs empty Modules where none exist yet if they are subpaths of target
def _assign_attr(from_obj: Any, to_module: flow.nn.Module, target: str):
    *prefix, field = target.split(".")
    for item in prefix:
        t = getattr(to_module, item, None)

        if t is None:
            t = flow.nn.Module()
            setattr(to_module, item, t)
        to_module = t

    # If it is a tensor and not a parameter attribute of a module, it should be a named buffer.
    # So, we register it as a named buffer in the target module.
    if isinstance(from_obj, flow.Tensor) and not isinstance(
        from_obj, flow.nn.Parameter
    ):
        to_module.register_buffer(field, from_obj)
    else:
        setattr(to_module, field, from_obj)


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class GraphModule(flow.nn.Module):
    """
    GraphModule is an nn.Module generated from an fx.Graph. Graphmodule has a
    ``graph`` attribute, as well as ``code`` and ``forward`` attributes generated
    from that ``graph``.

    .. warning::

        When ``graph`` is reassigned, ``code`` and ``forward`` will be automatically
        regenerated. However, if you edit the contents of the ``graph`` without reassigning
        the ``graph`` attribute itself, you must call ``recompile()`` to update the generated
        code.

    """

    def __new__(cls: "Type[GraphModule]", *args, **kwargs):
        # each instance of a graph module needs its own forward method
        # so create a new singleton class for each instance.
        # it is a subclass of the user-defined class, the only difference
        # is an extra layer to install the forward method

        class GraphModuleImpl(cls):  # type: ignore[misc, valid-type]
            pass

        return super().__new__(GraphModuleImpl)

    def __init__(
        self,
        root: Union[flow.nn.Module, Dict[str, Any]],
        graph: Graph,
        class_name: str = "GraphModule",
    ):
        """
        Construct a GraphModule.

        Args:

            root (Union[flow.nn.Module, Dict[str, Any]):
                ``root`` can either be an nn.Module instance or a Dict mapping strings to any attribute type.
                In the case that ``root`` is a Module, any references to Module-based objects (via qualified
                name) in the Graph's Nodes' ``target`` field will be copied over from the respective place
                within ``root``'s Module hierarchy into the GraphModule's module hierarchy.
                In the case that ``root`` is a dict, the qualified name found in a Node's ``target`` will be
                looked up directly in the dict's keys. The object mapped to by the Dict will be copied
                over into the appropriate place within the GraphModule's module hierarchy.

            graph (Graph): ``graph`` contains the nodes this GraphModule should use for code generation

            class_name (str): ``name`` denotes the name of this GraphModule for debugging purposes. If it's unset, all
                error messages will report as originating from ``GraphModule``. It may be helpful to set this
                to ``root``'s original name or a name that makes sense within the context of your transform.

        """
        super().__init__()
        self.__class__.__name__ = class_name
        if isinstance(root, flow.nn.Module):
            if hasattr(root, "training"):
                self.training = root.training
            for node in graph.nodes:
                if node.op in ["get_attr", "call_module"]:
                    assert isinstance(node.target, str)
                    _copy_attr(root, self, node.target)
        elif isinstance(root, dict):
            targets_to_copy = []
            for node in graph.nodes:
                if node.op in ["get_attr", "call_module"]:
                    assert isinstance(node.target, str)
                    if node.target not in root:
                        raise RuntimeError(
                            "Node "
                            + str(node)
                            + " referenced target "
                            + node.target
                            + " but that target was not provided in ``root``!"
                        )
                    targets_to_copy.append(node.target)
            # Sort targets in ascending order of the # of atoms.
            # This will ensure that less deeply nested attributes are assigned
            # before more deeply nested attributes. For example, foo.bar
            # will be assigned before foo.bar.baz. Otherwise, we might assign
            # the user-provided ``foo.bar`` and wipe out the previously-assigned
            # ``foo.bar.baz``
            targets_to_copy.sort(key=lambda t: t.count("."))
            for target_to_copy in targets_to_copy:
                _assign_attr(root[target_to_copy], self, target_to_copy)
        else:
            raise RuntimeError("Unsupported type " + str(root) + " passed for root!")

        self.graph = graph

        # Store the Tracer class responsible for creating a Graph separately as part of the
        # GraphModule state, except when the Tracer is defined in a local namespace.
        # Locally defined Tracers are not pickleable. This is needed because flow.package will
        # serialize a GraphModule without retaining the Graph, and needs to use the correct Tracer
        # to re-create the Graph during deserialization.
        self._tracer_cls = None
        if (
            self.graph._tracer_cls
            and "<locals>" not in self.graph._tracer_cls.__qualname__
        ):
            self._tracer_cls = self.graph._tracer_cls

    @property
    def graph(self) -> Graph:
        """
        Return the ``Graph`` underlying this ``GraphModule``
        """
        return self._graph

    @graph.setter
    def graph(self, g: Graph) -> None:
        """
        Set the underlying ``Graph`` for this ``GraphModule``. This will internally
        recompile the ``GraphModule`` so that the generated ``forward()`` function
        corresponds to ``g``
        """
        assert isinstance(g, Graph), f"Expected a Graph instance, but got {type(g)}"
        self._graph = g
        g.owning_module = self
        self.recompile()

    def to_folder(self, folder: Union[str, os.PathLike], module_name: str = "FxModule"):
        """Dumps out module to ``folder`` with ``module_name`` so that it can be
        imported with ``from <folder> import <module_name>``

        Args:

            folder (Union[str, os.PathLike]): The folder to write the code out to

            module_name (str): Top-level name to use for the ``Module`` while
                writing out the code
        """
        folder = Path(folder)
        Path(folder).mkdir(exist_ok=True)
        flow.save(self.state_dict(), folder / "state_dict.pt")
        tab = " " * 4
        model_str = f"""import oneflow as flow
from oneflow.nn import *
class {module_name}(flow.nn.Module):
    def __init__(self):
        super().__init__()
"""

        def _gen_model_repr(module_name: str, module: flow.nn.Module) -> Optional[str]:
            safe_reprs = [
                nn.Linear,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            ]
            if type(module) in safe_reprs:
                return f"{module.__repr__()}"
            else:
                return None

        blobified_modules = []
        for module_name, module in self.named_children():
            module_str = _gen_model_repr(module_name, module)
            if module_str is None:
                module_file = folder / f"{module_name}.pt"
                flow.save(module, module_file)
                blobified_modules.append(module_name)
                module_repr = module.__repr__().replace("\r", " ").replace("\n", " ")
                module_str = f"flow.load(r'{module_file}') # {module_repr}"
            model_str += f"{tab*2}self.{module_name} = {module_str}\n"

        for buffer_name, buffer in self._buffers.items():
            if buffer is None:
                continue
            model_str += f"{tab*2}self.register_buffer('{buffer_name}', flow.empty({list(buffer.shape)}))\n"

        for param_name, param in self._parameters.items():
            if param is None:
                continue
            model_str += f"{tab*2}self.{param_name} = flow.nn.Parameter(flow.empty({list(param.shape)}))\n"

        model_str += (
            f"{tab*2}self.load_state_dict(flow.load(r'{folder}/state_dict.pt'))\n"
        )
        model_str += f"{_addindent(self.code, 4)}\n"

        module_file = folder / "module.py"
        module_file.write_text(model_str)

        init_file = folder / "__init__.py"
        init_file.write_text("from .module import *")

        if len(blobified_modules) > 0:
            warnings.warn(
                "Was not able to save the following children modules as reprs -"
                f"saved as pickled files instead: {blobified_modules}"
            )

    def add_submodule(self, target: str, m: flow.nn.Module) -> bool:
        """
        Adds the given submodule to ``self``.

        This installs empty Modules where none exist yet if they are
        subpaths of ``target``.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``nn.Module.get_submodule`` for how to
                specify a fully-qualified string.)
            m: The submodule itself; the actual object we want to
                install in the current Module

        Return:
            bool: Whether or not the submodule could be inserted. For
                this method to return True, each object in the chain
                denoted by ``target`` must either a) not exist yet,
                or b) reference an ``nn.Module`` (not a parameter or
                other attribute)

        """
        *prefix, field = target.split(".")
        mod: flow.nn.Module = self

        for item in prefix:

            submod = getattr(mod, item, None)

            if submod is None:
                submod = flow.nn.Module()
                setattr(mod, item, submod)

            if not isinstance(submod, flow.nn.Module):
                return False

            mod = submod

        mod.add_module(field, m)
        return True

    def delete_all_unused_submodules(self) -> None:
        """
        Deletes all unused submodules from ``self``.

        A Module is considered "used" if any one of the following is
        true:
        1. It has children that are used
        2. Its forward is called directly via a ``call_module`` node
        3. It has a non-Module attribute that is used from a
        ``get_attr`` node

        This method can be called to clean up an ``nn.Module`` without
        manually calling ``delete_submodule`` on each unused submodule.
        """
        used: List[str] = []

        for node in self.graph.nodes:

            if node.op == "call_module" or node.op == "get_attr":

                # A list of strings representing the different parts
                # of the path. For exmaple, `foo.bar.baz` gives us
                # ["foo", "bar", "baz"]
                fullpath = node.target.split(".")

                # If we're looking at multiple parts of a path, join
                # join them with a dot. Otherwise, return that single
                # element without doing anything to it.
                def join_fn(x: str, y: str) -> str:
                    return ".".join([x, y] if y else [x])

                # Progressively collect all the names of intermediate
                # modules. For example, if we have the target
                # `foo.bar.baz`, we'll add `foo`, `foo.bar`, and
                # `foo.bar.baz` to the list.
                for path in itertools.accumulate(fullpath, join_fn):
                    used.append(path)

        to_delete = [name for name, _ in self.named_modules() if name not in used]

        for name in to_delete:
            self.delete_submodule(name)

    @property
    def code(self) -> str:
        """
        Return the Python code generated from the ``Graph`` underlying this
        ``GraphModule``.
        """
        if not hasattr(self, "_code"):
            raise RuntimeError(
                "Code has not been generated! Please report a bug to OneFlow"
            )
        return self._code

    def recompile(self) -> PythonCode:
        """
        Recompile this GraphModule from its ``graph`` attribute. This should be
        called after editing the contained ``graph``, otherwise the generated
        code of this ``GraphModule`` will be out of date.
        """
        if self._graph._pytree_info is not None:
            self._in_spec = self._graph._pytree_info.in_spec
            self._out_spec = self._graph._pytree_info.out_spec
        python_code = self._graph.python_code(root_module="self")
        self._code = python_code.src

        cls = type(self)
        cls.forward = _forward_from_src(self._code, python_code.globals)

        # Determine whether this class explicitly defines a __call__ implementation
        # to wrap. If it does, save it in order to have wrapped_call invoke it.
        # If it does not, wrapped_call can use a dynamic call to super() instead.
        # In most cases, super().__call__ should be torch.nn.Module.__call__.
        # We do not want to hold a reference to Module.__call__ here; doing so will
        # bypass patching of torch.nn.Module.__call__ done while symbolic tracing.
        cls_call = cls.__call__ if "__call__" in vars(cls) else None

        # Previously, if an error occurred when valid
        # symbolically-traced code was run with an invalid input, the
        # user would see the source of the error as coming from
        # `File "<eval_with_key_N">`, where N is some number. We use
        # this function to generate a more informative error message. We
        # return the traceback itself, a message explaining that the
        # error occurred in a traced Module's generated forward
        # function, and five lines of context surrounding the faulty
        # line
        def generate_error_message(frame_summary: traceback.FrameSummary) -> str:
            # auxiliary variables (for readability)
            err_lineno = frame_summary.lineno
            err_line_len = len(frame_summary.line)
            all_src_lines = linecache.getlines(frame_summary.filename)

            # constituent substrings of the error message
            tb_repr = traceback.format_exc()
            custom_msg = (
                "Call using an FX-traced Module, "
                f"line {err_lineno} of the traced Module's "
                "generated forward function:"
            )
            before_err = "".join(all_src_lines[err_lineno - 2 : err_lineno])
            marker = "~" * err_line_len + "~~~ <--- HERE"
            err_and_after_err = "\n".join(all_src_lines[err_lineno : err_lineno + 2])

            # joined message
            return "\n".join(
                [tb_repr, custom_msg, before_err, marker, err_and_after_err]
            )

        def wrapped_call(self, *args, **kwargs):
            try:
                if cls_call is not None:
                    return cls_call(self, *args, **kwargs)
                else:
                    return super(type(self), self).__call__(*args, **kwargs)
            except Exception as e:
                assert e.__traceback__
                topmost_framesummary: traceback.FrameSummary = traceback.StackSummary.extract(
                    traceback.walk_tb(e.__traceback__)
                )[
                    -1
                ]  # type: ignore[arg-type]
                if "eval_with_key" in topmost_framesummary.filename:
                    print(generate_error_message(topmost_framesummary), file=sys.stderr)
                raise e.with_traceback(None)

        cls.__call__ = wrapped_call

        return python_code
