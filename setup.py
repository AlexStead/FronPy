from setuptools import setup, find_packages

setup(
    name="fronpy",
    version="1.0.2",
    description="Stochastic frontier analysis in Python",
    author="Alexander D. Stead",
    author_email="a.d.stead@leeds.ac.uk",
    include_package_data=True,
    package_data={'fronpy': [
            'data/*.csv',
            'misc/*.ico',
            'misc/*.png',]},
    packages=find_packages(),
    install_requires=[
        "numpy>=1.25.2",
        "scipy>=1.11.1",
        "pandas>=2.2.3",
        "formulaic>=1.0.2",
        "statsmodels>=0.14.0",
        "mpmath>=1.3.0"
    ],
)