from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="catch-cap",
    version="0.1.4",
    author="axon_dendrite",
    author_email="amandogra2016@gmail.com",
    description="Detect and reduce LLM hallucinations with semantic entropy, log-probability analysis, and web grounding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adc77/catch-cap.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "google-genai>=0.5.0",
        "groq>=0.4.0",
        "tavily-python>=0.3.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black",
            "flake8",
            "mypy",
        ],
    },
)