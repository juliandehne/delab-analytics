# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="delab-analytics",
    version="0.0.1",
    description="a library to analyse social media conversations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juliandehne/delab-analytics",
    author="Julian Dehne & Valentin Gold",
    author_email="julian.dehne@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(),
    package_data={'delab-analytics.data': ['dataset_reddit_no_text.pkl', 'dataset_twitter_no_text.pkl']},
    include_package_data=False,
    install_requires=["numpy", "pandas", "networkx", "scikit-learn", "keras==2.11.0", "matplotlib",
                      "tensorflow==2.11.0"]
)
