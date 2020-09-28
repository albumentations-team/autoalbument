from setuptools import find_packages, setup

setup(
    name="autoalbument",
    version="0.0.1",
    install_requires=[
        "albumentations>=0.4.6",
        "torch>=1.6.0",
        "hydra-core>=1.0",
        "kornia>=0.4.0",
        "timm",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "autoalbument = autoalbument.cli.main:main",
            "autoalbument-search = autoalbument.cli.search:main",
        ],
    },
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
)
