# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""setup.py for ADHD."""

import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except OSError:
  README = ""

install_requires = [
    "numpy",
    "jax",
    "flax",
    "optax",
    "tensorstore",
    "ml-collections",
    "clu",
    "typing_extensions",
    "sentencepiece",
    "tensorflow_text>=2.4.0",
    "tensorflow_datasets",
    "tensorflow",
 ]

tests_require = [
    "pytest",
    "pytest-cov",
    "pytest-custom_exit_code",
    "pytest-xdist",
    "pytype",
]

__version__ = None

with open("adhd/version.py") as f:
  exec(f.read(), globals())

setup(
    name="adhd",
    version=__version__,
    description="ADHD: Attention-based modeling for those without it.",
    long_description="\n\n".join([README]),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="Jax team",
    author_email="jax-dev@google.com",
    url="https://github.com/levskaya/adhd",
    packages=find_packages(),
    package_data={"adhd": ["py.typed"]},
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        "testing": tests_require,
        },
    )


