import shutil
import tempfile
import platform
import os


def symlink(src, dst):
    if platform.system == "Windows":
        shutil.copy2(src, dst)
    else:
        tmp_dst = tempfile.mktemp(dir=dst.parents[0])
        os.symlink(src, tmp_dst)
        os.replace(tmp_dst, dst)
