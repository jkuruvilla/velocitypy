import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name="velocitypy",
    version="0.0.1",
    author="Joseph Kuruvilla",
    author_email="joseph.k@uni-bonn.de",
    description="Velocity statistics in cosmology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jkuruvilla/velocitypy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Topic :: Scientific/Engineering :: Astronomy",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    test_suite=tests,
)
