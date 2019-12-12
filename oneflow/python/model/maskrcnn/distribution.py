import functools
import oneflow

distribution_context = None


class DistributionContext(object):
    def __init__(self, num_devices_per_node, num_nodes, device_i, node_i):
        self.replicas = num_devices_per_node * num_nodes
        self.rank = node_i * num_devices_per_node + device_i
        self.device = device_i
        self.machine = node_i


def distribute_execute(num_devices_per_node, num_nodes, mode="train"):
    def decorator_func(func):
        @functools.wraps(func)
        def wrapped_func(config, *args, **kwargs):
            ret = []
            for node_i in range(num_nodes):
                for device_i in range(num_devices_per_node):
                    dist_ctx = DistributionContext(
                        num_devices_per_node, num_nodes, device_i, node_i
                    )
                    set_distribution_context(dist_ctx)
                    with oneflow.device_prior_placement(
                        "gpu", "{}:{}".format(node_i, device_i)
                    ):
                        with oneflow.experimental.distribution_name_scope(
                            node_i, device_i
                        ):
                            args_ = [arg[dist_ctx.rank] for arg in args]
                            kwargs_ = dict(
                                zip(
                                    kwargs.keys(),
                                    [v[dist_ctx.rank] for v in kwargs.values()],
                                )
                            )
                            ret.append(
                                func(dist_ctx, config, *args_, **kwargs_)
                            )
            if mode == "train":
                return tuple(map(list, zip(*ret)))
            elif mode == "eval":
                return ret
            else:
                raise NotImplementedError
        return wrapped_func
    return decorator_func


def set_distribution_context(ctx):
    global distribution_context
    distribution_context = ctx


def get_distribution_context():
    global distribution_context
    assert (
        distribution_context is not None
    ), "set distribution context before get"
    return distribution_context
