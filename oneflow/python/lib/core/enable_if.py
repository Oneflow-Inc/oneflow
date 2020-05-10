import traceback

def enable_if(bool_functor, default_func):
    assert default_func.__name__ == 'default'
    patterns_field = 'invoke_patterns__do_not_use_me_directly__'
    if not hasattr(default_func, patterns_field): setattr(default_func, patterns_field, [])
    invoke_patterns = getattr(default_func, patterns_field)
    location = _GetFrameLocationStr(-2)
    def get_failed_info(customized_prompt=None):
        failed_info = "no avaliable function found.\n" 
        for bf, func, location in invoke_patterns:
            if customized_prompt is None:
                prompt = "\n%s: \033[1;31mFAILED\033[0m" % location
            else:
                prompt = customized_prompt
            failed_info += "\n%s \n\t%s\n" % (prompt, bf.debug_str())
        return failed_info
    def decorator(func):
        assert func.__name__ == 'invoke'
        invoke_patterns.append((bool_functor, func, location))
        def wrapper(*args, **kwargs):
            select_func = None
            for bool_functor, func, _ in invoke_patterns:
                if bool_functor():
                    assert select_func is None 
                    select_func = func
            if select_func is not None: return select_func(*args, **kwargs)
            return default_func(get_failed_info, *args, **kwargs)
        wrapper.__is_specialization_supported__ = True
        return wrapper
    return decorator

def _GetFrameLocationStr(depth = -1):
    assert depth < 0
    frame = traceback.extract_stack()[depth -1]
    return "%s:%d" % (frame[0], frame[1])

