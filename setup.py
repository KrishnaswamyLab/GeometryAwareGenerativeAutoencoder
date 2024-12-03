from setuptools import setup, find_packages

setup(
    name='dmae',
    version='0.1.0',
    author='Xingzhi Sun',
    author_email='xingzhi.sun@yale.edu',
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/xingzhis/dmae',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'hydra-core',
        'wandb',
        'pytorch-lightning',
        'torch',
    ],
    # entry_points={
    #     'console_scripts': [
    #         'my_script=dmae.file1:main_function',  # Replace with your script and function
    #     ],
    # },
)
