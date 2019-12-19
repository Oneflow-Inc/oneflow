import functools


class VariableGetterComposite(object):
    def __init__(self):
        self.getter_stack = []

    def __call__(self, var_gen_fn, *args, **kwargs):
        def make_inner(outter, inner):
            @functools.wraps(inner)
            def inner_fn():
                return outter(inner, *args, **kwargs)

            return inner_fn

        fn = var_gen_fn
        for getter in self.getter_stack:
            fn = make_inner(getter, fn)

        return fn()

    def register(self, fn):
        self.getter_stack.append(fn)
