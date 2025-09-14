from setuptools import setup, find_packages

setup(
    name="contextq",
    version="0.1.4",
    description="ContextQ: Activation aware LLM Quantization, Make any LLM stick to your tasks",
    author="Ayan Jhunjhunwala",
    author_email="ayanqwerty@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers>=4.51",
        "datasets",
        "accelerate",
        "awq"
        
    ],
    entry_points={
        "console_scripts": [
            "contextq=contextq.awq_profiles:main",
        ],
    }
)
