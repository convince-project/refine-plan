from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='refine_plan',
   version='0.0.1',
   description='Automatic Behaviour Tree Refinement',
   long_description=long_description,
   author='Charlie Street',
   author_email='c.l.street@bham.ac.uk',
   packages=['refine_plan'],
   package_dir = {'': 'src'},
   install_requires=['pyeda', 'stormpy']
)
