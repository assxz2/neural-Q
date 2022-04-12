# -*- coding: utf-8 -*-
# @Time     : 2019/07/16 20:13
# @Author   : Bill H
# @Email    : lcurious@163.com
# @File     : setup.py
# @License  : Apache-2.0
# Copyright (C) Bill H All rights reserved
import setuptools

with open("README.md", "r", encoding='UTF-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="rouran",
    version="0.0.1",
    author="Huang Zenan",
    author_email="lccurious@outlook.com",
    description="Iterative registration toolbox for animal videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lccurious/rouran",
    install_requires=[],
    packages=setuptools.find_packages(),
    data_files=[('rouran/constants', ['rouran/constants/template_config.yaml',
                                       'rouran/constants/landmark_hrnet_config.yaml']),
                ('rouran/object_tracking/snapshot',
                 ['rouran/object_tracking/snapshot/CIResNet22_RPN.pth']),
                ('rouran/landmark_estimation/pretrained',
                 ['rouran/landmark_estimation/pretrained/hrnetv2_w18_imagenet_pretrained.pth'])],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points="""[console_scripts]
            rc=rc:main""",
)

# https://stackoverflow.com/questions/39590187/in-requirements-txt-what-does-tilde-equals-mean
