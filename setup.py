from __future__ import absolute_import
import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ImbalancedLearningRL", 
    version="1",
    author="Jenny Yang",
    author_email="jenny.yang@eng.ox.ac.uk",
    description="ImbalancedLearningRL provides automatic training of a classifier with outcome label imbalances.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/yangjenny/ImbalancedLearningRL/",
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: MIT License',
        'Operating System :: OS Independent'
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        'tensorflow',
        'numpy',
        'pandas',
        'sklearn',
        'shap'
    ]
)