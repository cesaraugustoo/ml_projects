from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scientific_discovery",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for scientific discovery using LLMs and graph analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scientific_discovery",
    packages=find_packages(where=".", exclude=["tests*"]),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core Python utilities
        "typing>=3.7.4",
        "pathlib>=1.0.1",
        "requests>=2.28.0",
        "tqdm>=4.65.0",
        "python-dateutil>=2.8.2",
        
        # Data processing and analysis
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        
        # Deep Learning and NLP
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "langchain>=0.0.200",
        "openai>=1.0.0",
        
        # Graph Processing
        "networkx>=2.8.0",
        "pyvis>=0.3.0",
        
        # Visualization
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "pillow>=9.0.0",  # PIL fork
        
        # Jupyter and Display
        "ipython>=8.0.0",
        "jupyter>=1.0.0",
        "ipywidgets>=8.0.0",
        
        # Utility packages
        "pyyaml>=6.0",
        "python-dotenv>=0.20.0",  # for environment variables
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=1.0.0',
            'isort>=5.0.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'myst-parser>=1.0.0',
        ],
    },
entry_points={
        'console_scripts': [
            'scientific_discovery=scientific_discovery.src.cli:main',
        ],
    },
    package_data={
        'scientific_discovery': ['src/data/*.json'],
    },
    include_package_data=True,
)