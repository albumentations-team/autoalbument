import io
import os
import re

from setuptools import find_packages, setup


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "autoalbument", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


setup(
    name="autoalbument",
    version=get_version(),
    install_requires=[
        "albumentations @ https://github.com/albumentations-team/albumentations/archive/updated_transforms.zip#egg=albumentations-0.4.7",  # noqa: E501
        "torch>=1.6.0",
        "hydra-core>=1.0",
        "timm",
        "tqdm",
        "click",
        "colorama",
        "tensorboard",
    ],
    entry_points={
        "console_scripts": [
            "autoalbument-create = autoalbument.cli.create:main",
            "autoalbument-search = autoalbument.cli.search:main",
        ],
    },
    extras_require={"tests": ["pytest"]},
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
)
