from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

exec(open('src/clustfeatimp/version.py').read())

setup(
    name="clustfeatimp",
    version=__version__,
    author="Mateusz Soczewka",
    author_email="msoczewkas@gmail.com",
    description="Module for measuring the feature importance in clustering models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/msoczi/clustfeatimp",
    project_urls={
        "Bug Tracker": "https://github.com/msoczi/clustfeatimp/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.4",
        "scikit-learn>=1.1.2",
        "xgboost>=1.6.2",
        "seaborn>=0.12.0",
        "matplotlib>=3.4.3",
        "scikit-optimize>=0.9.0"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
)
