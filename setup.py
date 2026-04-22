from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).resolve().parent
long_description = (here / "README.rst").read_text(encoding="utf-8")

version_ns = {}
exec((here / "BGlib" / "__version__.py").read_text(encoding="utf-8"), version_ns)
__version__ = version_ns["version"]

# TODO: Move requirements to requirements.txt
requirements = [
    "numpy>=1.13.0",
    "h5py>=2.6.0",
    "scipy>=0.17.1",
    "scikit-image>=0.12.3",
    "scikit-learn>=0.17.1",
    "matplotlib>=2.0.0",
    "psutil",
    "six",
    "pillow",
    "joblib>=0.11.0",
    "ipywidgets>=5.2.2",
    "ipython>=6.0",
    "numpy_groupies>=0.9.22",  # New build of 0.9.8.4 appears to cause build problems
    "sidpy>=0.12.1",
    "pyUSID>=0.0.11",
    "xlrd>=1.0.0",
]

setup(
    name="BGlib",
    version=__version__,
    description="Band Excitation and General Mode analysis and visualization codes",
    long_description=long_description,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords=["imaging", "spectra", "multidimensional", "scientific"],
    py_modules=["bglib_mcp_server"],
    python_requires=">=3.10",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    url="https://pycroscopy.github.io/BGlib/about.html",
    license="MIT",
    author="Suhas Somnath, Chris R. Smith, Rama K. Vasudevan, Stephen Jesse, Anton Ievlev, and contributors",
    author_email="pycroscopy@gmail.com",
    install_requires=requirements,
    platforms=["Linux", "Mac OSX", "Windows 10/8.1/8/7"],
    include_package_data=True,
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-dependencies
    extras_require={
        "mcp": ["mcp>=1.0.0"],
    },
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    # package_data={
    #     'sample': ['package_data.dat'],
    # },
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        "console_scripts": [
            "bglib-mcp=bglib_mcp_server:main",
        ],
    },
)
