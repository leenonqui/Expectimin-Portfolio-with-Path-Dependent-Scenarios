from setuptools import setup, find_packages

setup(
    name="expectiminimax-portfolio",
    version="0.1.0",
    description="Portfolio optimization using GIC methodology and expectiminimax framework",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pydantic>=1.8.0",
        "jupyterlab",
        "matplotlib",
        "seaborn"
    ],
    python_requires=">=3.8",
)
