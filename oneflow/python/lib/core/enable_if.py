import oneflow.python.lib.core.traceinfo as traceinfo

def enable_if(*args):
    conditional_functions = [(hob_expr, func, func.__name__) for func, hob_expr in args]
    matched_func = GetMatchedFunction(conditional_functions)
    if matched_func is not None: return matched_func
    def default(get_failed_info, *args, **kwargs):
        raise NotImplementedError(get_failed_info())
    return MakeDefaultFunction(default, conditional_functions)


def GetMatchedFunction(conditional_functions):
    select_triple = (None, None, None)
    for triple in conditional_functions:
        if not triple[0](): continue
        if select_triple[1] is not None: return _MultiMatchedErrorFunction([select_triple, triple])
        select_triple = triple
    return select_triple[1]

def MakeDefaultFunction(default, conditional_functions):
    def get_failed_info(customized_prompt=None):
        failed_info = "no avaliable function found.\n" 
        for bf, func, location in conditional_functions:
            prompt = location if customized_prompt is None else customized_prompt
            failed_info += "\n%s: \033[1;31mFAILED\033[0m\n\t%s\n" % (prompt, bf.debug_str())
        return failed_info
    return lambda *args, **kwargs: default(get_failed_info, *args, **kwargs)

def _MultiMatchedErrorFunction(matched_functions):
    def raise_assert_error(*args, **kwargs):
        failed_info = "at least two conditional functions matched.\n" 
        for bf, func, location in matched_functions:
            failed_info += "\n%s: \033[1;31mPASSED\033[0m\n\t%s\n" % (location, bf.debug_str())
        raise AssertionError(failed_info)
    return raise_assert_error
