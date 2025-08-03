from setuptools import setup, find_packages

setup(
    name="contextq",
    version="0.1.2",
    description="ContextQ: Context Based adjustments for LLMs with attention and quantization",
    author="Ayan Jhunjhunwala",
    author_email="ayanqwerty@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers>=4.43",
        "bitsandbytes>=0.43",
        "datasets",
        "accelerate",
    ],
    entry_points={
        "console_scripts": [
            "selective-gpt=contextq.main:main",
        ],
    }
)