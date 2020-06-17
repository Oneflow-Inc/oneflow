import inspect
import sys

if sys.version_info > (2, 7) and sys.version_info < (3, 0):

    def GetArgNameAndDefaultTuple(func):
        """
      returns a dictionary of arg_name:default_values for the input function
      """
        args, varargs, keywords, defaults = inspect.getargspec(func)
        defaults = list(defaults) if defaults is not None else []
        while len(defaults) < len(args):
            defaults.insert(0, None)
        return tuple(zip(args, defaults))


elif sys.version_info >= (3, 0):

    def GetArgNameAndDefaultTuple(func):
        signature = inspect.signature(func)
        return tuple(
            [
                (k, v.default if v.default is not inspect.Parameter.empty else None)
                for k, v in signature.parameters.items()
            ]
        )


else:
    raise NotImplementedError


def GetArgDefaults(func):
    return tuple(map(lambda x: x[1], GetArgNameAndDefaultTuple(func)))
