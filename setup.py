from setuptools import find_packages, setup

setup(
    name="autoalbument",
    version="0.0.1",
    install_requires=[
        "albumentations @ https://github.com/albumentations-team/albumentations/archive/updated_transforms.zip#egg=albumentations-0.4.7",  # noqa: E501
        "torch>=1.6.0",
        "hydra-core>=1.0",
        "timm",
        "tqdm",
        "click",
        "colorama",
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
