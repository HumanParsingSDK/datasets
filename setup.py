import setuptools
import human_datasets


with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as req_file:
    requirements = req_file.readlines()


setuptools.setup(
    name="human_datasets",
    version=human_datasets.__version__,
    author="Anton Fedotov",
    author_email="anton.fedotov.af@gmail.com",
    description="Utils for human parsing datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HumanParsingSDK/datasets",
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
