from setuptools import setup, find_packages

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path):
    requirements = []

    with open(file_path) as file_obj:
        for line in file_obj:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line == HYPHEN_E_DOT:
                continue

            requirements.append(line)

    return requirements

setup(
    name="crypto_time_series_analysis",
    version="0.1.0",
    author="Mohammad Arshad",
    author_email="siddiquiarshad106@gmail.com",
    description="End-to-End Machine Learning Pipeline for Cryptocurrency Time Series Forecasting and Causality Analysis",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.10",
)