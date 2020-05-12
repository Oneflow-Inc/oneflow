import oneflow.python.lib.core.traceinfo as traceinfo

def enable_if(bool_functor, default):
    assert default.__name__ == 'default'
    patterns_field = 'invoke_patterns__do_not_use_me_directly__'
    if not hasattr(default, patterns_field): setattr(default, patterns_field, [])
    conditional_functions = getattr(default, patterns_field)
    location = traceinfo.GetFrameLocationStr(-2)
    default_func = MakeDefaultFunction(default, conditional_functions)
    def decorator(func):
        assert func.__name__ == 'invoke'
        conditional_functions.append((bool_functor, func, location))
        def wrapper(*args, **kwargs):
            matched_func = GetMatchedFunction(conditional_functions)
            if matched_func is None: matched_func = default_func
            return matched_func(*args, **kwargs)
        wrapper.__is_specialization_supported__ = True
        return wrapper
    return decorator

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
