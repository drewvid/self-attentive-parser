import setuptools
import sys


setuptools.setup(
    name="benepar",
    version="0.2.1",
    author="Nikita Kitaev",
    author_email="kitaev@cs.berkeley.edu",
    description="Berkeley Neural Parser",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nikitakit/self-attentive-parser",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.9",
    classifiers=(
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ),
    install_requires=[
        "nltk>=3.9.4",
        "spacy>=3.8.14",
        "torch>=2.12.0",
        "torch-struct @ git+https://github.com/harvardnlp/pytorch-struct@7146de5659ff17ad7be53023c025ffd099866412",
        "tokenizers>=0.22.2",
        "transformers[torch,tokenizers]>=5.10.2",
        "protobuf>=7.35.0",
        "sentencepiece>=0.2.1",
        "dataclasses;python_version<'3.7'",
    ],
)
