"""Setup configuration for ASL Hand Sign Detection package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="asl-detection",
    version="1.0.0",
    author="Paran Kafle",
    author_email="your.email@uts.edu.au",
    description="ASL Hand Sign Detection using deep learning and skeleton-guided approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ParanKafleUTS/ASL",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "asl-train=src.training.trainer:main",
            "asl-evaluate=src.evaluation.metrics:main",
            "asl-download=data.download_dataset:main",
        ],
    },
)
