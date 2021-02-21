from setuptools import setup

setup(name='gym_randomwalk',
    version='0.0.1',
    install_requires=['gym'],
    author="Lukas Schwarz",
    author_email="code@lukasschwarz.de",
    description="A 1d random walk gym environment",
    packages=setuptools.find_packages(),
)
