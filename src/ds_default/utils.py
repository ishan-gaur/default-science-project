import os
from patlib import Path
from DEFAULT_PKG_CHANGE.constants import HOME_FOLDER, DATA_FOLDER, DB_FOLDER, OUTPUT_FOLDER

import inspect

def optional_import(pkg_name, install_cmd=None):
    """
    GPT Generated
    Import a package with a clear, context-aware error message if it's missing.
    
    Args:
        pkg_name (str): The package to import, e.g., 'torch'.
        install_cmd (str, optional): Custom install command to show the user.
                                      Defaults to 'pip install <pkg_name>'.
    Returns:
        module: The imported package.

    Raises:
        ImportError: If the package isn't installed, with a clear message
                     referencing the calling function.
    """
    try:
        return __import__(pkg_name)
    except ImportError as e:
        # Get name of the function that triggered this import
        stack = inspect.stack()
        calling_function = None
        for frame in stack[1:]:
            func_name = frame.function
            if func_name != "<module>":  # skip top-level module calls
                calling_function = func_name
                break

        if install_cmd is None:
            install_cmd = f"pip install {pkg_name}"

        location_msg = (
            f" while running `{calling_function}()`"
            if calling_function
            else ""
        )

        raise ImportError(
            f"The package `{pkg_name}` is required{location_msg} but not installed.\n"
            f"Install it with:\n\n    {install_cmd}\n"
        ) from e


def save_tensor_to_sacred(t, ex, name, data_partition=Path.cwd()):
    if not isinstance(Path, data_partition):
        data_partition = Path(data_partition)
    filepath = data_partition / f"tensor_{name}_{ex.current_run.id}.pt"
    torch.save(t, filepath)
    ex.add_artifact(filepath)
    os.remove(filepath)
