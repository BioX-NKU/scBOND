# scBOND: Biologically faithful bidirectional translation between single-cell transcriptomes and DNA methylomes with adaptability to paired data scarcity 

A sophisticated framework for bidirectional cross-modality translation between scRNA-seq and scDNAm profiles with broad biological applicability. We show that **scBOND**(a) accurately translates data while preserving biologically significant differences between closely related cell types. It also recovers functional and tissue-specific signals in the human brain and reveals stage-specific and cell type-specific transcriptional-epigenetic mechanisms in the oligodendrocyte lineage. We further introduce **scBOND-Aug**, a powerful enhancement of scBOND that leverages biologically guided data augmentation, achieving remarkable performance and surpassing traditional methods in paired data-limited scenarios.

![fig](https://github.com/BioX-NKU/scBOND/blob/main/figures/main.png)

## Installation

It's prefered to create a new environment for scBOND

```
conda create -n scBond python==3.9
conda activate scBond
```

scBOND is available on PyPI, and could be installed using

```
pip install scBond
```

Installation via Github is also provided

```
git clone https://github.com/Biox-NKU/scBOND
cd scBOND
pip install -r requirements.txt
```

This process will take approximately 5 to 10 minutes, depending on the user's computer device and internet connectivition.

## Quick Start

Illustrating with the translation between  scRNA-seq and scDNAm data as an example, scBOND could be easily used following 3 steps: data preprocessing, model training, predicting and evaluating. 

Generate a scBOND model first with following process:

```python
from scBond.bond import Bond
bond = Bond()
```

### 1. Data preprocessing

* Before data preprocessing, you should load the **raw count matrix** of scRNA-seq and scDNAm data via `bond.load_data`:
  
  ```python
  bond.load_data(RNA_data, MET_data, train_id, test_id, validation_id)
  ```
  
  | Parameters    | Description                                                  |
  | ------------- | ------------------------------------------------------------ |
  | RNA_data      | AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes. |
  | MET_data      | AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to peaks. |
  | train_id      | A list of cell IDs for training.                             |
  | test_id       | A list of cell IDs for testing.                              |
  | validation_id | An optional list of cell IDs for validation, if setted None, bond will use a default setting of 20% cells in train_id. |
  
  Anndata object is a Python object/container designed to store single-cell data in Python packege [**anndata**](https://anndata.readthedocs.io/en/latest/) which is seamlessly integrated with [**scanpy**](https://scanpy.readthedocs.io/en/stable/), a widely-used Python library for single-cell data analysis.

* For data preprocessing, you could use `bond.data_preprocessing`:
  
  ```python
  bond.data_preprocessing()
  ```
  
  You could save processed data or output process logging to a file using following parameters.
  
  | Parameters   | Description                                                                                  |
  | ------------ | -------------------------------------------------------------------------------------------- |
  | save_data    | optional, choose save the processed data or not, default False.                              |
  | file_path    | optional, the path for saving processed data, only used if `save_data` is True, default None.  |
  | logging_path | optional, the path for output process logging, if not save, set it None, default None.       |

  scBOND also support to refine this process using other parameter, however, we strongly recommend the default settings to keep the best result for model.
  
  <details>
  <summary>other parameters</summary>
      <table border="1" style="border-collapse: collapse; width: 100%;"> <thead> <tr> <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Parameter</th> <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Description</th> </tr> </thead> <tbody> <tr> <td style="padding: 8px;"><code>normalize_total</code></td> <td style="padding: 8px;">Whether to normalize total RNA expression per cell, default True</td> </tr> <tr> <td style="padding: 8px;"><code>log1p</code></td> <td style="padding: 8px;">Whether to apply log(1+x) transformation to RNA data, default True</td> </tr> <tr> <td style="padding: 8px;"><code>use_hvg</code></td> <td style="padding: 8px;">Whether to select highly variable genes, default True</td> </tr> <tr> <td style="padding: 8px;"><code>n_top_genes</code></td> <td style="padding: 8px;">Number of highly variable genes to select, default 3000</td> </tr> <tr> <td style="padding: 8px;"><code>imputation</code></td> <td style="padding: 8px;">Imputation method for missing values in methylation data ('median' or other), default 'median'</td> </tr> <tr> <td style="padding: 8px;"><code>min_cells</code></td> <td style="padding: 8px;">Filter out features present in fewer than this fraction of cells, default 0.007</td> </tr> <tr> <td style="padding: 8px;"><code>normalize</code></td> <td style="padding: 8px;">Normalization method for methylation data ('scale' or other), default 'scale'</td> </tr> <tr> <td style="padding: 8px;"><code>add_noise</code></td> <td style="padding: 8px;">Whether to add Gaussian noise to data, default False</td> </tr> <tr> <td style="padding: 8px;"><code>noise_rate</code></td> <td style="padding: 8px;">Standard deviation of Gaussian noise to add, default 0.0</td> </tr> <tr> <td style="padding: 8px;"><code>noise_seed</code></td> <td style="padding: 8px;">Random seed for noise generation, default 42</td> </tr> <tr> <td style="padding: 8px;"><code>save_data</code></td> <td style="padding: 8px;">Whether to save processed data, default False</td> </tr> <tr> <td style="padding: 8px;"><code>file_path</code></td> <td style="padding: 8px;">Path for saving processed data (if save_data is True), default None</td> </tr> </tbody> </table>
### 2. Model training

* Before model training, you could choose to use data augmentation strategy or not. If using data augmentation, scBOND will generate synthetic samples with the use of cell-type labels(if `cell_type` in `adata.obs`) .

  scButterfly provide data augmentation API:
  
  ```python
  bond.augmentation(enable_augmentation)
  ```

  You could choose parameter `enable_augmentation` by whether you want to augment data (`True`) or not (`False`), this will cause more training time used, but promise better result for predicting. 
  
  * If you choose `enable_augmentation = True`, scBOND-Aug will try to find `cell_type` in `adata.obs`. If failed, it will automaticly transfer to `False`.
  * If you just want to using original data for scBOND training, set `enable_augmentation = False`.
  
* You could construct a scBOND model as following:
  
  ```python
  bond.construct_model(chrom_list)
  ```
  
  scBOND need a list of peaks count for each chromosome, remember to sort peaks with chromosomes.
  
  | Parameters   | Description                                                                                    |
  | ------------ | ---------------------------------------------------------------------------------------------- |
  | chrom_list   | a list of peaks count for each chromosome, remember to sort peaks with chromosomes.            |
  | logging_path | optional, the path for output model structure logging, if not save, set it None, default None. |
  
* scBOND model could be easily trained as following:
  
  ```python
  bond.train_model()
  ```

  | Parameters   | Description                                                                             |
  | ------------ | --------------------------------------------------------------------------------------- |
  | output_path  | optional, path for model check point, if None, using './model' as path, default None.   |
  | load_model   | optional, the path for load pretrained model, if not load, set it None, default None.   |
  | logging_path | optional, the path for output training logging, if not save, set it None, default None. |
  
  scBOND also support to refine the model structure and training process using other parameters for `bond.construct_model()` and `bond.train_model()` .
  
  <details>
  <summary>other parameters for model construction</summary>
      <table border="1" style="border-collapse: collapse; width: 100%;"> <thead> <tr> <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Parameter</th> <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Description</th> </tr> </thead> <tbody> <tr> <td style="padding: 8px;"><code>R_encoder_nlayer</code></td> <td style="padding: 8px;">Layer counts of RNA encoder, default 2</td> </tr> <tr> <td style="padding: 8px;"><code>M_encoder_nlayer</code></td> <td style="padding: 8px;">Layer counts of methylation data encoder, default 2</td> </tr> <tr> <td style="padding: 8px;"><code>R_decoder_nlayer</code></td> <td style="padding: 8px;">Layer counts of RNA decoder, default 2</td> </tr> <tr> <td style="padding: 8px;"><code>M_decoder_nlayer</code></td> <td style="padding: 8px;">Layer counts of methylation data decoder, default 2</td> </tr> <tr> <td style="padding: 8px;"><code>R_encoder_dim_list</code></td> <td style="padding: 8px;">Dimension list of RNA encoder, length equal to R_encoder_nlayer, default [256, 128]</td> </tr> <tr> <td style="padding: 8px;"><code>M_encoder_dim_list</code></td> <td style="padding: 8px;">Dimension list of methylation data encoder, length equal to M_encoder_nlayer, default [32, 128]</td> </tr> <tr> <td style="padding: 8px;"><code>R_decoder_dim_list</code></td> <td style="padding: 8px;">Dimension list of RNA decoder, length equal to R_decoder_nlayer, default [128, 256]</td> </tr> <tr> <td style="padding: 8px;"><code>M_decoder_dim_list</code></td> <td style="padding: 8px;">Dimension list of methylation data decoder, length equal to M_decoder_nlayer, default [128, 32]</td> </tr> <tr> <td style="padding: 8px;"><code>R_encoder_act_list</code></td> <td style="padding: 8px;">Activation list of RNA encoder, length equal to R_encoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()]</td> </tr> <tr> <td style="padding: 8px;"><code>M_encoder_act_list</code></td> <td style="padding: 8px;">Activation list of methylation data encoder, length equal to M_encoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()]</td> </tr> <tr> <td style="padding: 8px;"><code>R_decoder_act_list</code></td> <td style="padding: 8px;">Activation list of RNA decoder, length equal to R_decoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()]</td> </tr> <tr> <td style="padding: 8px;"><code>M_decoder_act_list</code></td> <td style="padding: 8px;">Activation list of methylation data decoder, length equal to M_decoder_nlayer, default [nn.LeakyReLU(), nn.Sigmoid()]</td> </tr> <tr> <td style="padding: 8px;"><code>translator_embed_dim</code></td> <td style="padding: 8px;">Dimension of embedding space for translator, default 128</td> </tr> <tr> <td style="padding: 8px;"><code>translator_input_dim_r</code></td> <td style="padding: 8px;">Dimension of input from RNA encoder for translator, default 128</td> </tr> <tr> <td style="padding: 8px;"><code>translator_input_dim_m</code></td> <td style="padding: 8px;">Dimension of input from methylation data encoder for translator, default 128</td> </tr> <tr> <td style="padding: 8px;"><code>translator_embed_act_list</code></td> <td style="padding: 8px;">Activation list for translator, involving [mean_activation, log_var_activation, decoder_activation], default [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()]</td> </tr> <tr> <td style="padding: 8px;"><code>discriminator_nlayer</code></td> <td style="padding: 8px;">Layer counts of discriminator, default 1</td> </tr> <tr> <td style="padding: 8px;"><code>discriminator_dim_list_R</code></td> <td style="padding: 8px;">Dimension list of discriminator for RNA, length equal to discriminator_nlayer, default [128]</td> </tr> <tr> <td style="padding: 8px;"><code>discriminator_dim_list_M</code></td> <td style="padding: 8px;">Dimension list of discriminator for methylation data, length equal to discriminator_nlayer, default [128]</td> </tr> <tr> <td style="padding: 8px;"><code>discriminator_act_list</code></td> <td style="padding: 8px;">Activation list of discriminator, length equal to discriminator_nlayer, default [nn.Sigmoid()]</td> </tr> <tr> <td style="padding: 8px;"><code>dropout_rate</code></td> <td style="padding: 8px;">Rate of dropout for network, default 0.1</td> </tr> <tr> <td style="padding: 8px;"><code>R_noise_rate</code></td> <td style="padding: 8px;">Rate of setting part of RNA input data to 0, default 0.5</td> </tr> <tr> <td style="padding: 8px;"><code>M_noise_rate</code></td> <td style="padding: 8px;">Rate of setting part of methylation data input to 0, default 0.3</td> </tr> <tr> <td style="padding: 8px;"><code>num_experts</code></td> <td style="padding: 8px;">Number of experts for translator, default 6</td> </tr> <tr> <td style="padding: 8px;"><code>num_experts_single</code></td> <td style="padding: 8px;">Number of experts for single translator, default 6</td> </tr> <tr> <td style="padding: 8px;"><code>num_heads</code></td> <td style="padding: 8px;">Number of parallel attention heads, default 8</td> </tr> <tr> <td style="padding: 8px;"><code>attn_drop</code></td> <td style="padding: 8px;">Dropout probability applied to attention weights, default 0.1</td> </tr> <tr> <td style="padding: 8px;"><code>proj_drop</code></td> <td style="padding: 8px;">Dropout probability applied to the output projection, default 0.1</td> </tr> </tbody> </table>
  
  <details>
  <summary>other parameters for model training</summary>
      <table border="1" style="border-collapse: collapse; width: 100%;"> <thead> <tr> <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Parameter</th> <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Description</th> </tr> </thead> <tbody> <tr> <td style="padding: 8px;"><code>R_encoder_lr</code></td> <td style="padding: 8px;">Learning rate of RNA encoder, default 0.001</td> </tr> <tr> <td style="padding: 8px;"><code>M_encoder_lr</code></td> <td style="padding: 8px;">Learning rate of methylation data encoder, default 0.001</td> </tr> <tr> <td style="padding: 8px;"><code>R_decoder_lr</code></td> <td style="padding: 8px;">Learning rate of RNA decoder, default 0.001</td> </tr> <tr> <td style="padding: 8px;"><code>M_decoder_lr</code></td> <td style="padding: 8px;">Learning rate of methylation data decoder, default 0.001</td> </tr> <tr> <td style="padding: 8px;"><code>R_translator_lr</code></td> <td style="padding: 8px;">Learning rate of RNA pretrain translator, default 0.0001</td> </tr> <tr> <td style="padding: 8px;"><code>M_translator_lr</code></td> <td style="padding: 8px;">Learning rate of methylation data pretrain translator, default 0.0001</td> </tr> <tr> <td style="padding: 8px;"><code>translator_lr</code></td> <td style="padding: 8px;">Learning rate of translator, default 0.0001</td> </tr> <tr> <td style="padding: 8px;"><code>discriminator_lr</code></td> <td style="padding: 8px;">Learning rate of discriminator, default 0.005</td> </tr> <tr> <td style="padding: 8px;"><code>R2R_pretrain_epoch</code></td> <td style="padding: 8px;">Max epoch for pretrain RNA autoencoder, default 100</td> </tr> <tr> <td style="padding: 8px;"><code>M2M_pretrain_epoch</code></td> <td style="padding: 8px;">Max epoch for pretrain methylation data autoencoder, default 100</td> </tr> <tr> <td style="padding: 8px;"><code>lock_encoder_and_decoder</code></td> <td style="padding: 8px;">Lock the pretrained encoder and decoder or not, default False</td> </tr> <tr> <td style="padding: 8px;"><code>translator_epoch</code></td> <td style="padding: 8px;">Max epoch for train translator, default 200</td> </tr> <tr> <td style="padding: 8px;"><code>patience</code></td> <td style="padding: 8px;">Patience for loss on validation, default 50</td> </tr> <tr> <td style="padding: 8px;"><code>batch_size</code></td> <td style="padding: 8px;">Batch size for training and validation, default 64</td> </tr> <tr> <td style="padding: 8px;"><code>r_loss</code></td> <td style="padding: 8px;">Loss function for RNA reconstruction, default nn.MSELoss(size_average=True)</td> </tr> <tr> <td style="padding: 8px;"><code>m_loss</code></td> <td style="padding: 8px;">Loss function for methylation data reconstruction, default nn.BCELoss(size_average=True)</td> </tr> <tr> <td style="padding: 8px;"><code>d_loss</code></td> <td style="padding: 8px;">Loss function for discriminator, default nn.BCELoss(size_average=True)</td> </tr> <tr> <td style="padding: 8px;"><code>loss_weight</code></td> <td style="padding: 8px;">List of loss weight for [r_loss, a_loss, d_loss], default [1, 2, 1]</td> </tr> <tr> <td style="padding: 8px;"><code>seed</code></td> <td style="padding: 8px;">Set up the random seed, default 19193</td> </tr> <tr> <td style="padding: 8px;"><code>kl_mean</code></td> <td style="padding: 8px;">Size average for kl divergence or not, default True</td> </tr> <tr> <td style="padding: 8px;"><code>R_pretrain_kl_warmup</code></td> <td style="padding: 8px;">Epoch of linear weight warm up for kl divergence in RNA pretrain, default 50</td> </tr> <tr> <td style="padding: 8px;"><code>M_pretrain_kl_warmup</code></td> <td style="padding: 8px;">Epoch of linear weight warm up for kl divergence in methylation data pretrain, default 50</td> </tr> <tr> <td style="padding: 8px;"><code>translation_kl_warmup</code></td> <td style="padding: 8px;">Epoch of linear weight warm up for kl divergence in translator pretrain, default 50</td> </tr> </tbody> </table>
### 3. Predicting and evaluating

* scBOND provide a predicting API, you could get predicted profiles as follow:

  ```python
  M2R_predict, R2M_predict = bond.test_model()
  ```

  A series of evaluating method also be integrated in this function, you could get these evaluation using parameters:

  | Parameters    | Description                                                                                 |
  | ------------- | ------------------------------------------------------------------------------------------- |
  | output_path   | optional, path for model evaluating output, if None, using './model' as path, default None. |
  | load_model    | optional, the path for load pretrained model, if not load, set it None, default False.      |
  | model_path    | optional, the path for pretrained model, only used if `load_model` is True, default None.   |
  | test_cluster  | optional, test the correlation evaluation or not, including **AMI**, **ARI**, **HOM**, **NMI**, default False.|
  | test_figure   | optional, draw the **tSNE** visualization for prediction or not, default False.             |
  | output_data   | optional, output the prediction to file or not, if True, output the prediction to `output_path/A2R_predict.h5ad` and `output_path/R2A_predict.h5ad`, default False.                                          |

- Also, scBOND provide **a separate predicting API** for single modal predicting. You can predict DNAm profile with RNA profile as follow:

  ```python
  R2M_predict = bond.predict_single_modal(data_type='rna')
  ```

  And you can predict RNA profile with DNAm profile as follow:

  ```python
  M2R_predict = bond.predict_single_modal(data_type='met')
  ```

## Demo, document, tutorial and source code

### We provide demos of basic scBOND model and two variants (scBOND_Aug and one for single modality prediction) with GSE140493 dataset in [scBOND usage](https://github.com/BioX-NKU/scBOND/blob/main/example/scBOND_usage.ipynb), [scBOND-aug usage](https://github.com/BioX-NKU/scBOND/blob/main/example/scBOND_aug_usage.ipynb) and  [scBOND for single modality prediction](https://github.com/BioX-NKU/scBOND/blob/main/example/scBOND_for_single_modal.ipynb).

### We also provide richer tutorials and documents for scBOND in [scBOND documents](http://scbond.readthedocs.io), including more details of provided APIs for customing data preprocessing, model structure and training strategy.

