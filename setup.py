from setuptools import setup

setup(name="nninit",
      author="Alykhan Tejani",
      author_email="alykhan.tejani@gmail.com",
      description="initialization schemes for the PyTorch nn package",
      version="0.1.dev1",
      url="https://github.com/alykhantejani/nninit",
      py_modules=["nninit"],
      install_requires=['torch', 'numpy', 'scipy'])