from setuptools import setup

setup(name='gym_gridworld',
    version='0.0.1',
    install_requires=['gym'],
    author="Lukas Schwarz",
    author_email="code@lukasschwarz.de",
    description="A Gym environment representing a 2d rectangular grid world",
    packages=setuptools.find_packages(),
)
