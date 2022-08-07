from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent

README = (HERE/"readme.md").read_text()

setup(
    name= "pytorchsummary",
    version="1.0.3",
    description="Summary of PyTorch Models just like `model.summary() in Keras",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/GSAUC3/pytorch-model-details",
    author="Rajarshi Banerjee",
    author_email="raju.banerjee.720@gmail.com",
    license="MIT",
    keywords=['python','PyTorch','Pytorch model summary','Pytorch parameter summary'],
    packages=["pytorchsummary"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)