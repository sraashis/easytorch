import pathlib

from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="quenn",
    version="1.133",
    description="Quick Neural Network Experimentation",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/sraashis/nnkernel",
    author="Aashis Khana1",
    author_email="sraashis@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=['quenn', 'quenn.utils', 'quenn.core'],
    include_package_data=True,
    install_requires=['setuptools', 'wheel', 'numpy', 'scipy', 'scikit-learn', 'scikit-image',
                      'pillow', 'matplotlib', 'torch', 'torchvision',
                      'opencv-python', 'pandas', 'seaborn']
)
