import tempfile
import os


def symlink(src, dst):
    tmp_dst = tempfile.mktemp(dir=dst.parents[0])
    os.symlink(src, tmp_dst)
    os.replace(tmp_dst, dst)
