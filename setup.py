from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="qsttoolkit", 
    version="1.0.0", 
    description="The definitive, open-source package for performing optical quantum state tomography using both traditional (statistical) and novel deep learning-powered methods in Python.", 
    long_description=long_description, 
    long_description_content_type="text/markdown", 
    author="George FitzGerald", 
    author_email="gwfitzg@hotmail.com", 
    url="https://github.com/georgefitzgerald02/qst-toolkit",
    license="MIT",
    packages=find_packages(), 
    install_requires=[
        "numpy==1.26.4",
        "scipy==1.13.1",
        "pandas==2.2.2",
        "matplotlib==3.10.0",
        "seaborn==0.13.2",
        "qutip==5.1.1",
        "scikit-learn==1.6.0",
        "tensorflow==2.17.1"
    ], 
    classifiers=[
        "Programming Language :: Python :: 3",  # Python 3 compatibility
        "License :: OSI Approved :: MIT License",  # License type
        "Operating System :: OS Independent",  # Cross-platform compatibility
    ], 
    include_package_data=True,  # Include additional files like README and LICENSE in the package
)
