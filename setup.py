"""
DUNE-Ω∞-1900-L Setup
"""
from setuptools import setup, find_packages

setup(
    name="dune-omega",
    version="1.0.0",
    description="DUNE-Ω∞-1900-L: Distributed Unified Neuro-Emergent Intelligence",
    long_description=open("README.md", "r").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="DUNE Research",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    extras_require={
        "test": ["pytest", "unittest"],
        "dev": ["pytest", "black", "mypy"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)