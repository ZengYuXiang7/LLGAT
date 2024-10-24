from setuptools import setup, find_packages

setup(
    name="LLGAT",
    version="0.1.0",
    author="Yuxiang Zeng",
    author_email="zengyuxiang@hnu.edu.cn",
    description="A deep learning project focusing on GAT",
    long_description_content_type="text/markdown",
    url="https://github.com/Zengyuxiang7/LLGAT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy==1.24.4",
        "pandas==2.0.3",
        "scipy==1.10.1",
        "scikit-learn==1.3.2",
        "openpyxl==3.1.5",
        "node2vec==0.4.6",
        "pysnooper==1.2.0",
        "loguru",
        "einops",
        "nbformat",
        "numexpr>=2.8.0",
        "yagmail>=0.14.0",
        "faiss-cpu==1.6.5",
        "openai==0.28.0",
        "netron",
    ],
    entry_points={
        "console_scripts": [
            "run=Run_demo:main",  # 将 `run` 函数注册为命令行工具
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.rst", "*.md"],
        "your_package": ["data/*.dat"],
    },
)