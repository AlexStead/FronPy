from setuptools import setup, find_packages

setup(
    name="fronpy",
    version="0.0.0",
    description="Stochastic frontier analysis in Python",
    author="Alexander D. Stead",
    author_email="a.d.stead@leeds.ac.uk",
    include_package_data=True,
    package_data={'fronpy': ['data/*.csv']},
    packages=find_packages(),
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "statsmodels>=0.14.0"
    ],
)