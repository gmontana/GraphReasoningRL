from setuptools import setup, find_packages

setup(
    name="deeppath",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
    ],
    author="DeepPath Team",
    author_email="",
    description="Reinforcement Learning for Knowledge Graph Reasoning",
    keywords="knowledge-graph, reinforcement-learning, path-finding",
    url="",
    python_requires=">=3.8",
)