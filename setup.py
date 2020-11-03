import io
import os
import re

from setuptools import find_packages, setup


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "autoalbument", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


setup(
    name="autoalbument",
    version=get_version(),
    description="AutoML for image augmentation",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Alex Parinov, Vladimir Iglovikov, Eugene Khvedchenya, Druzhinin Mikhail, Buslaev Alexander",
    license="MIT",
    url="https://github.com/albumentations-team/autoalbument",
    install_requires=[
        "albumentations>=0.5.1",
        "torch>=1.6.0",
        "hydra-core>=1.0",
        "timm==0.1.20",  # This version is required for segmentation-models-pytorch
        "segmentation-models-pytorch",
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
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
