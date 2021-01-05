import setuptools

with open("README.md", 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="sampling",
    version="0.1",
    description="bayesian adaptive sampling with label reuse",
    url="https://github.com/factil/sampling.git",
    author="Factil",
    author_email="ran.xiao@factil.io",
    install_requires=["numpy==1.19.0",
                      "scipy==1.6.0"
    ],
    python_requires='>=3.7'
)
