from setuptools import setup, find_packages

setup(
    name="tumor-growth-rbf",
    version="0.2.0",
    author="Robert Fuge",
    author_email="rephug@gmail.com",
    description="Meshless tumor growth simulator using RBF-FD",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
)
