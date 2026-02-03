"""Setup configuration for drowsiness-detection package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drowsiness-detection",
    version="1.0.0",
    author="Tim",
    description="Comparing neural networks vs. engineered features for drowsiness detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tpatzelt/drowsiness-detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib~=3.5.1",
        "numpy~=1.21.6",
        "pandas~=1.3.5",
        "scikit-learn~=1.0.2",
        "tensorflow~=2.8.0",
        "scipy~=1.7.3",
        "click~=8.0.3",
        "prettytable~=3.0.0",
        "sacred~=0.8.2",
        "sklearn-evaluation~=0.5.7",
        "dill~=0.3.5.1",
        "ConfigSpace~=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.10.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
        ],
    },
)
