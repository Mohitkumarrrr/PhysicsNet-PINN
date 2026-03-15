"""
Setup script for PhysicsNet-PINN.

Install in development mode:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="physicsnet-pinn",
    version="1.0.0",
    author="Mohit Kumar",
    author_email="mohit.kumar@example.com",
    description=(
        "A lightweight Python library implementing Physics-Informed Neural "
        "Networks (PINNs) for solving partial differential equations."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/PhysicsNet-PINN",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "matplotlib>=3.5",
        "scipy>=1.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "flake8>=5.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="physics-informed neural-network PDE deep-learning",
)
