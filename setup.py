from setuptools import setup, find_packages

setup(
    name="TokenSim",
    version="0.1",
    packages=find_packages(include=["TokenSim", "TokenSim.*", "util", "util.*"]),
)

