import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name='REL',
    version='0.0.1',
    # scripts=['dokr'] ,
    author="Johannes Michael",
    author_email="mick.vanhulst@gmail.com",
    description="Entity Linking package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mickvanhulst/rel",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    install_requires=required,
    include_package_data=True,
    python_requires='>=3.6',
)
