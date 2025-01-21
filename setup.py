from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="qsttoolkit", 
    version="0.2.2", 
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
        "pandas==2.2.2",
        "scipy==1.13.1",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "qutip==5.0.4",
        "scikit-learn==1.5.1",
        "tensorflow==2.10.0"
    ], 
    classifiers=[
        "Programming Language :: Python :: 3",  # Python 3 compatibility
        "License :: OSI Approved :: MIT License",  # License type
        "Operating System :: OS Independent",  # Cross-platform compatibility
    ], 
    python_requires=">=3.9", 
    # test_suite="tests",  # Location of tests - not ready yet
    include_package_data=True,  # Include additional files like README and LICENSE in the package
)
