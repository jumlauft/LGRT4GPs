# $ python setup.py sdist bdist_wheel
# $ twine upload dist/*
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.rst").read_text()

version = {}
with open("./lgrt4gps/version.py") as fp:
    exec(fp.read(), version)

# This call to setup() does all the work
setup(
    name="lgrt4gps",
    version=version['__version__'],
    description="Locally growing random trees for Gaussian processes",
    long_description=README,
  #  long_description_content_type="text/markdown",
    url="https://github.com/jumlauft/LGRT4GPs",
    author="Jonas Umlauft",
    author_email="jumlauft@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["numpy",
                      "scipy",
                      "paramz",
                      "gpy",
                      "six",
                      "binarytree",
                      "matplotlib"],
    entry_points={
        "console_scripts": [
            "realpython=tests.test_lgrt:main",
        ]
    },
)