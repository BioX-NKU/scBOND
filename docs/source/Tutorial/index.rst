
Tutorial
========

Translation between single-cell transcriptomes and DNA methylomes
-----------------------------------------------------------------

scBOND could make translation between single-cell transcriptomes and DNA methylomes paired data using scBOND model in ``scBond.bond``.

* `scBOND usage <RNA_DNAm_paired_basic/scBOND_usage.ipynb>`_

Extension usages of scBOND framework
-----------------------------------------

scBOND provide a series of extension usage with preprocessing in ``scBond.data_processing`` 
and model in ``scBond.train_model`` (scRNA-seq ~ scDNAm data)

Translation based on data augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Based on cell-type labels, scBOND could make translation after data augmentation strategy ``scBond.bond.Bond.augmentation``.

* `scBOND_Aug using augmentation with cell-type labels <RNA_DNAm_variants/scBOND_aug_usage.ipynb>`_

Translation from single-modal data to another modality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After training, scBOND could use single scRNA-seq to predict single scDNAm in ``scBond.train_model.Model.predict_single_methylation``
and use single scDNAm to predict single scRNA-seq in ``scBond.train_model.Model.predict_single_rna``.

* `scBOND for single modality prediction <RNA_DNAm_variants/scBOND_for_single_model.ipynb>`_

