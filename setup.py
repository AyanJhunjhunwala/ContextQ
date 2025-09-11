from setuptools import setup, find_packages

setup(
    name="contextq",
    version="0.1.3",
    description="ContextQ: Context Based adjustments for LLMs with attention and quantization",
    author="Ayan Jhunjhunwala",
    author_email="ayanqwerty@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers>=4.51",
        "datasets",
        "accelerate",
        "autoawq",
        "autoawq-kernels",
    ],
    entry_points={
        "console_scripts": [
            "contextq=contextq.awq_profiles:main",
        ],
    }
)
