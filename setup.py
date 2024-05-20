from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="refine_plan",
    version="1.0.0",
    description="Automatic Behaviour Tree Refinement",
    long_description=long_description,
    author="Charlie Street",
    author_email="c.l.street@bham.ac.uk",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["pyeda", "stormpy", "numpy", "sympy"],
)
