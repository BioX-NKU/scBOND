from setuptools import setup, find_packages
import pathlib
import os

here = pathlib.Path(__file__).parent.resolve()
version = {}
version_file = os.path.join(os.path.dirname(__file__), "scBond", "version.py")
with open(version_file) as f:
    exec(f.read(), version)
    
setup(
    name="scBond",
    version=version["__version__"],
    description="A single-cell cross-modality translation method specifically between RNA data and DNA methylation data.",
    long_description="A sophisticated framework for bidirectional cross-modality translation between scRNA-seq and scDNAm profiles with broad biological applicability. We show that scBOND accurately translates data while preserving biologically significant differences between closely related cell types. It also recovers functional and tissue-specific signals in the human brain and reveals stage-specific and cell type-specific transcriptional-epigenetic mechanisms in the oligodendrocyte lineage. ",
    license="MIT Licence",
    author="PiperL",
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    keywords="single-cell multi-omics; cross-modality translation; single-cell RNA sequencing; single-cell DNA methylation; transcriptional-epigenetic regulation",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        'scanpy>=1.9.1',
        'torch>=1.12.1',
        'torchvision>=0.13.1',
        'torchaudio>=0.12.1',
        'scikit-learn>=1.1.3',
        'scvi-tools==0.19.0',
        'scvi-colab',
        'scipy==1.9.3',
        'episcanpy==0.3.2',
        'seaborn>=0.11.2',
        'matplotlib>=3.6.2',
        'pot==0.9.0',
        'torchmetrics>=0.11.4',
        'leidenalg',
        'pybedtools',
        'adjusttext',
        'jupyter'
    ]
)