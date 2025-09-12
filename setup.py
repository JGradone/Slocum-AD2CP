from setuptools import find_packages, setup

setup(
    name="Slocum-AD2CP",  # distribution name (what pip sees)
    version="0.1.0",
    description="Processing and analysis of TWR Slocum Glider-Nortek AD2CP data",
    author="Joe Gradone",
    license="MIT",
    packages=find_packages(where="src"),   # look inside src/
    package_dir={"": "src"},               # root is src/
    install_requires=[
        # add dependencies here, e.g. "numpy", "xarray"
    ],
    python_requires=">=3.8",
)
