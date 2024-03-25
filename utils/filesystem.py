# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import shutil
import os
import stat

def copyfile(src: Path, dst: Path, overwrite: bool=False):
    """Copies a file using shutil.copy2 but also adjusts write permission if necessary and overwrite is True."""
    src = Path(src)
    dst = Path(dst)

    if dst.exists() and not overwrite:
        return

    if dst.exists() and overwrite:
        mode = stat.S_IMODE(os.stat(dst).st_mode)
        if (mode & stat.S_IWUSR) == 0:
            os.chmod(dst, mode | stat.S_IWUSR)
    return shutil.copy2(src, dst)