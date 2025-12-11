|PyPI| |Docs| |PyPIDownloads|


.. |PyPI| image:: https://img.shields.io/pypi/v/scbond
   :target: https://pypi.org/project/scbond/
.. |Docs| image:: https://readthedocs.org/projects/scbond/badge/?version=latest
   :target: https://scbond.readthedocs.io
.. |PyPIDownloads| image:: https://pepy.tech/badge/scbond
   :target: https://pepy.tech/project/scbond



scBOND: Biologically faithful bidirectional translation between single-cell transcriptomes and DNA methylomes with adaptability to paired data scarcity
==================================================================================================================

Single-cell multi-omics sequencing technologies have offered unprecedented insights into cellular heterogeneity by jointly profiling gene expression and epigenetic landscapes at single-cell resolution. However, the application of these technologies remains limited due to technical challenges and high costs. Computational approaches for cross-modality translation provide a promising solution to these limitations by enabling the inference of one modality from another. However, existing methods for cross-modality translation between single-cell RNA sequencing (scRNA-seq) and single-cell DNA methylation (scDNAm) data face limitations, including unidirectionality, inadequate modeling of context-specific DNA methylation-expression associations, neglect of biological relevance in evaluation, and poor performance in limited paired training data. To fill these gaps, we introduce scBOND, a bidirectional cross-modal translation framework tailored for scRNA-seq and scDNAm profiles. scBOND leverages a Mixture-of-Experts block to capture context-dependent regulatory patterns, while implementing self-attention mechanism and a feature recalibration module to enhance biological signal fidelity. Extensive experiments demonstrate scBOND consistently outperforms baseline methods in both translation directions, yielding high-accuracy translation while preserving cellular structure. In mouse embryonic data, scBOND preserves subtle, functionally significant differences between closely related cell types, which are undetected in original data. Downstream analyses confirmed that scBOND effectively recovers tissue-specific signals in human brain neurons. Moreover, using RNA-only data, we reconstructed scDNAm profiles and identified cell type- and stage-specific regulatory mechanisms in oligodendrocyte lineage. To further improve model generalization in paired data-scarce scenarios, we propose scBOND-Aug, a variant of scBOND equipped with a biologically informed data augmentation strategy, which demonstrates superior results with limited paired data.

.. toctree::
   :maxdepth: 2
   :hidden:
   
   API/index
   Tutorial/index
   Installation
   release/index


News
----

.. include:: news.rst
   :start-line: 2
   :end-line: 22  
