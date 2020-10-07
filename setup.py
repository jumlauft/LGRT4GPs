import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="LGRT4GPs",
    version="1.0.0",
    description="Locally growing random trees for Gaussian processes",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jumlauft/LGRT4GPs",
    author="Jonas Umlauft",
    author_email="jumlauft@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["reader"],
    include_package_data=True,
    install_requires=["numpy", "scipy", "gpy", "matplotlib"],
    entry_points={
        "console_scripts": [
            "realpython=tests.test_lgrt:main",
        ]
    },
)