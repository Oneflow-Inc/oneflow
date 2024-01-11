import os


def parse_boolean_from_env(env_var, default_value):
    env_var = os.getenv(env_var)
    if env_var is None:
        return default_value
    env_var = env_var.lower()
    return env_var in ("1", "true", "yes", "on", "y")


def set_boolean_env_var(env_var: str, val: bool):
    os.environ[env_var] = str(val)
