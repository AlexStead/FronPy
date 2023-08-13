from setuptools import setup, find_packages

setup(
    name="FronPy",
    version="0.0.0",
    description="Stochastic frontier analysis in Python",
    author="Alexander D. Stead",
    author_email="a.d.stead@leeds.ac.uk",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "statsmodels>=1.0",
        "pkg_resources>=0.14.0"
    ],
)