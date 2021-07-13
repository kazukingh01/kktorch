from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kktorch*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kktorch',
    version='1.0.0',
    description='pytorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazukingh01/kktorch",
    author='kazuking',
    author_email='kazukingh01@gmail.com',
    license='Public License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Private License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'absl-py==0.13.0',
        'cachetools==4.2.2',
        'certifi==2021.5.30',
        'chardet==4.0.0',
        'entmax==1.0',
        'google-auth==1.32.1',
        'google-auth-oauthlib==0.4.4',
        'grpcio==1.38.1',
        'idna==2.10',
        'Markdown==3.3.4',
        'more-itertools==8.8.0',
        'numpy==1.21.0',
        'oauthlib==3.1.1',
        'opencv-python==4.5.3.56'
        'Pillow==8.3.1',
        'protobuf==3.17.3',
        'pyasn1==0.4.8',
        'pyasn1-modules==0.2.8',
        'pytorch-ranger==0.1.1',
        'requests==2.25.1',
        'requests-oauthlib==1.3.0',
        'rsa==4.7.2',
        'six==1.16.0',
        'tensorboard==2.5.0',
        'tensorboard-data-server==0.6.1',
        'tensorboard-plugin-wit==1.8.0',
        'torch-optimizer==0.1.0',
        'torchvision==0.10.0',
        'typing-extensions==3.10.0.0',
        'urllib3==1.26.6',
        'Werkzeug==2.0.1',
    ],
    python_requires='>=3.7'
)