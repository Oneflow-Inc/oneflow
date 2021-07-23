_function_docstr = {}

def add_docstr(fun, docstr: str):
    _function_docstr[fun] = docstr

def register_docstr():
    for (fun, docstr) in _function_docstr.items():
        setattr(fun, '__doc__', docstr)