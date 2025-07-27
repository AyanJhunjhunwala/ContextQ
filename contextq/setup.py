from setuptools import setup, find_packages
setup(
    name="contextq",
    version="0.1",
    description="ContextQ: Context Based adjustments for LLMs with attention and quantization",
    author="Ayan Jhunjhunwala",
    author_email="ayanqwerty@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch",
    ]
)
