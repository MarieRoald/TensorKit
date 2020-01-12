from setuptools import setup, find_packages


setup(
    name="TensorKit",
    version="0.0.002a",
    packages=find_packages("src"),
    package_dir={"": "src"}
)
