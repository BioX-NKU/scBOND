
Tutorial
========

Translation between single-cell transcriptomes and DNA methylomes
-----------------------------------------------------------------

scBOND could make translation between single-cell transcriptomes and DNA methylomes paired data using scBOND model in ``scBond.bond``.

* `scBOND usage <https://github.com/BioX-NKU/scBOND/blob/main/examples/scBOND_usage.ipynb>`_
* `scBOND_Aug using augmentation with cell-type labels <https://github.com/BioX-NKU/scBOND/blob/main/examples/scBOND_aug_usage.ipynb>`_

Extension usages of scBOND framework
-----------------------------------------

scBOND provide a series of extension usage with preprocessing in ``scBond.data_processing`` 
and model in ``scBond.train_model`` (scRNA-seq ~ scDNAm data)

Translation between scRNA-seq and scDNAm unpaired data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After training, scBOND could use single scRNA-seq to predict single scDNAm in ``scBond.train_model.Model.predict_single_methylation``
and use single scDNAm to predict single scRNA-seq in ``scBond.train_model.Model.predict_single_rna``.

* `Unpaired data translation using <https://github.com/BioX-NKU/scBOND/blob/main/examples/scBOND_for_single_modal.ipynb>`_


.. toctree::
    :maxdepth: 2
    :hidden:

    scBOND_usage
    scBOND_aug_usage
    scBOND_for_single_modal
