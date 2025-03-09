from setuptools import setup, find_packages

setup(
    name="celeris",
    version="0.1",
    packages=find_packages(),
    description="Celeris: A GPU library for rapid matrix operations.",
    install_requires=["numpy", "torch"],
    include_package_data=True,
) 