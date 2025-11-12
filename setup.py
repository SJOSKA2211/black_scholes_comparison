from setuptools import setup, find_packages

setup(
    name='black_scholes_comparison',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        # e.g., 'numpy>=1.24.0',
        # These are already in requirements.txt, but can be duplicated here
        # for proper package distribution.
    ],
    python_requires='>=3.10',
    author='Your Name',
    author_email='your.email@example.com',
    description='A comparative analysis of numerical methods for Black-Scholes PDE',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/black_scholes_comparison',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
