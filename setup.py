from setuptools import find_packages, setup

setup(
    name="slocum-ad2cp",  # 🔑 PyPI package name (must be lowercase, hyphens allowed)
    version="2.0.0",
    description="Processing and analysis of TWR Slocum Glider-Nortek AD2CP data",
    author="Joe Gradone",
    license="MIT",
    packages=find_packages(where="src"),   # look inside src/
    package_dir={"": "src"},               # root is src/
    install_requires=[
        "numpy",
        "xarray",
        "pandas",
        "netCDF4",
        "scipy",
        "gsw",
        "dask",  # required by xarray.open_mfdataset in load_ad2cp
    ],
    extras_require={
        # Glider data backends. The AD2CP processing itself needs neither.
        "erddap": ["erddapy"],
        "dbd": ["dbdreader>=0.6", "lz4"],  # >=0.6 reads compressed dcd/ecd
        "all": ["erddapy", "dbdreader>=0.6", "lz4"],
    },
    python_requires=">=3.9",  # keep consistent with pyproject.toml
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/JGradone/Slocum-AD2CP",
    project_urls={
        "Bug Tracker": "https://github.com/JGradone/Slocum-AD2CP/issues",
    },
)
