import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras-one_cycle_lr",
    version="0.0.1",
    author="Shivam Agarwal",
    author_email="shivam.agarwal151@gmail.com",
    description="Keras implementation of One Cycle Policy and LR Finder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shivam-agarwal-17/Keras-training-tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
