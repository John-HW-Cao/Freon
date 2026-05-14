from setuptools import setup

setup(
    name="freon-optimizer",
    version="0.1.0",
    description=(
        "Freon and Kaon: Schatten quasi-norm optimizers from "
        "'Muon is Not That Special: Random or Inverted Spectra Work Just as Well' "
        "(arXiv:2605.11181)"
    ),
    py_modules=["freon"],
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
)
