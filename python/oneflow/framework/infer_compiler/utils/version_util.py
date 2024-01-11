from importlib_metadata import version
from .log_utils import logger

def get_support_message():
    recipient_email = "caishenghang@oneflow.org"

    message = f"""\033[91m Advanced features cannot be used !!! \033[0m
If you need unrestricted multiple resolution, quantization support or any other more advanced features, please send an email to \033[91m{recipient_email}\033[0m and tell us about your use case, deployment scale and requirements.
        """
    return message


def is_quantization_enabled():
    import oneflow

    if version("oneflow") < "0.9.1":
        RuntimeError(
            "onediff_comfy_nodes requires oneflow>=0.9.1 to run.", get_support_message()
        )
        return False
    try:
        import diffusers_quant
    except ImportError as e:
        logger.warning(
            f"Failed to import diffusers_quant, Error message: {e}, {get_support_message()}"
        )
        return False
    return hasattr(oneflow._C, "dynamic_quantization")


def is_community_version():
    is_community = not is_quantization_enabled()
    return is_community
