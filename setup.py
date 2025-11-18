from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="face-enhancement-ai",
    version="1.0.0",
    author="Security Camera AI Team",
    description="State-of-the-art face image enhancement for security camera footage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexv879/Security-Camera-Face-Image-Enhancement",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "facexlib>=0.3.0",
        "basicsr>=1.4.2",
        "gfpgan>=1.3.8",
        "click>=8.1.0",
        "rich>=13.5.0",
        "fastapi>=0.103.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "api": [
            "fastapi>=0.103.0",
            "uvicorn[standard]>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "face-enhance=face_enhancement.cli:main",
            "face-enhance-server=face_enhancement.api.server:main",
        ],
    },
)
