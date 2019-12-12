from __future__ import absolute_import

import functools
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.placement_util as placement_util

from oneflow.python.framework.remote_blob import RemoteBlob
from oneflow.python.advanced.distribute_split import distribute_split
from oneflow.python.advanced.distribute_clone import distribute_clone
from oneflow.python.deprecated.variable_scope import distribution_name_scope
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("experimental.mirror_execute")
def mirror_execute(devices_per_node, nodes):
    if compile_context.cur_job_distribution_strategy is None:
        compile_context.cur_job_distribution_strategy = MirrorDistributionStrategy(
            devices_per_node, nodes
        )

    def decorator_fn(fn):
        return compile_context.cur_job_distribution_strategy.register(fn)

    return decorator_fn


def dictzip(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(dct[i] for dct in dcts)


class DistributionStrategy(object):
    pass


class MirrorDistributionStrategy(DistributionStrategy):
    def __init__(self, devices_per_node, nodes):
        self.devices_per_node = devices_per_node
        self.nodes = nodes
        self.mirror_variables = {}
        self.cur_rank = -1

    def rank(self, node_i, device_i):
        return self.devices_per_node * node_i + device_i

    def save_placement_scope(self):
        cur_placement_scope = placement_util.cur_placement_scope()
        self.original_device_tag = cur_placement_scope.default_device_tag
        self.original_machine_device_ids = cur_placement_scope.machine_device_ids_

    def register_variable_getter(self):
        @compile_context.variable_getter
        def get_mirror_variable(var_gen_fn, name, *args, **kwargs):
            if name not in self.mirror_variables:
                with placement_util.DevicePriorPlacementScope(
                    self.original_device_tag, self.original_machine_device_ids
                ):
                    self.mirror_variables[name] = distribute_clone(
                        var_gen_fn(), name=name + "_DistributeClone"
                    )

            return self.mirror_variables[name][self.cur_rank]

    def register(self, fn):
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            args_ = [distribute_split(arg) if isinstance(arg, RemoteBlob) else arg for arg in args]
            args_is_blob = [True if isinstance(arg, RemoteBlob) else False for arg in args]

            kwargs_ = dict(
                zip(
                    kwargs.keys(),
                    [
                        distribute_split(v) if isinstance(v, RemoteBlob) else v
                        for v in kwargs.values()
                    ],
                )
            )
            kwargs_is_blob = dict(
                zip(
                    kwargs.keys(),
                    [True if isinstance(v, RemoteBlob) else False for v in kwargs.values()],
                )
            )

            self.save_placement_scope()
            self.register_variable_getter()

            result = []
            for node_i in range(self.nodes):
                for device_i in range(self.devices_per_node):
                    self.cur_rank = self.rank(node_i, device_i)
                    with placement_util.DevicePriorPlacementScope(
                        self.original_device_tag, "{}:{}".format(node_i, device_i)
                    ):
                        with distribution_name_scope(node_i, device_i):
                            args__ = [
                                arg[self.cur_rank] if is_blob else arg
                                for arg, is_blob in zip(args_, args_is_blob)
                            ]
                            kwargs__ = dict(
                                [
                                    (k, v[self.cur_rank]) if is_blob else (k, v)
                                    for k, v, is_blob in dictzip(kwargs_, kwargs_is_blob)
                                ]
                            )
                            result.append(fn(*args__, **kwargs__))

            return result

        return wrapped_fn
