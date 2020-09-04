from setuptools import setup, find_packages

setup(
    name="covid_moonshot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.1",
        "dataclasses-json>=0.5",
        "pymbar>=3.0",
        "fire>=0.3",
        "jinja2>=2.11",
        "joblib>=0.16",
        "simplejson>=3.17",
        "mdtraj>=1.9.4",
        "matplotlib>=3.3",
        "seaborn>=0.10",
        "rich>=6.0"
    ],
    entry_points={"console_scripts": ["covid-moonshot = covid_moonshot.app:main"]},
    author="Matt Wittmann",
    author_email="matt.wittmann@choderalab.org",
    description="Tools and infrastructure for automating COVID Moonshot analysis",
    url="https://github.com/choderalab/covid-moonshot-infra",
)
