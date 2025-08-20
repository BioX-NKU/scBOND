import getopt
import sys
import gc
import os
from scBond.data_processing import RNA_data_preprocessing, MET_data_preprocessing
import scanpy as sc
import anndata as ad
from scBond.logger import *
from scBond.model_utlis import *
from scBond.calculate_cluster import *
from scBond.split_datasets import *
import warnings
import torch
import torch.nn as nn
from scBond.train_model import Model

warnings.filterwarnings("ignore")

class Bond():
    def __init__(
        self,
     ):
        """
        scBond model.

        """
        setup_seed(19193)
        self.my_logger = create_logger(name='Bond', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
    
    def load_data(
        self,
        RNA_data,
        MET_data,
        train_id,
        test_id,
        validation_id = None
     ):
        """
        Load data to Bond model.
        
        Parameters
        ----------
        train_id: list
            list of cell ids for training.
            
        test_id: list
            list of cell ids for testing.
            
        validation_id: list
            list of cell ids for validation, if None, Butterfly will use default 20% train cells for validation.

        """
        self.RNA_data = RNA_data.copy()
        self.MET_data = MET_data.copy()
        self.test_id_r = test_id.copy()
        self.test_id_m = test_id.copy()
        if validation_id is None:
            self.train_id = train_id
            random.shuffle(self.train_id)
            train_count = int(len(self.train_id)*0.8)
            self.train_id_r = self.train_id[0:train_count].copy()
            self.train_id_m = self.train_id[0:train_count].copy()
            self.validation_id_r = self.train_id[train_count:].copy()
            self.validation_id_m = self.train_id[train_count:].copy()
            del self.train_id
        else:
            self.train_id_r = train_id.copy()
            self.train_id_m = train_id.copy()
            self.validation_id_r = validation_id.copy()
            self.validation_id_m = validation_id.copy()
        self.my_logger.info('successfully load in data with\n\nRNA data:\n'+str(RNA_data)+'\n\nMET data:\n'+str(MET_data))
        self.is_processed = False
          
        
    def load_data_unpaired(
        self,
        RNA_data,
        MET_data,
        train_id_r=None,
        train_id_m=None,
        validation_id_r=None,
        validation_id_m=None,
        test_id_r=None,
        test_id_m=None,
        train_id=None,
        test_id=None,
        validation_id=None
    ):
        """
        Load data to Bond model.

        Parameters
        ----------
        RNA_data: AnnData
            RNA expression data.

        MET_data: AnnData
            Methylation data.

        train_id_r: list
            list of cell ids for RNA training.

        train_id_a: list
            list of cell ids for methylation data training.

        validation_id_r: list
            list of cell ids for RNA validating.

        validation_id_a: list
            list of cell ids for methylation data validating.

        test_id_r: list
            list of cell ids for RNA testing.

        test_id_a: list
            list of cell ids for methylation data testing.
        """
        self.RNA_data = RNA_data.copy()
        self.MET_data = MET_data.copy()

        self.train_id_r = train_id_r
        self.train_id_a = train_id_m
        self.validation_id_r = validation_id_r
        self.validation_id_a = validation_id_m
        self.test_id_r = test_id_r
        self.test_id_a = test_id_m

        self.my_logger.info('successfully load in data with\n\nRNA data:\n'+str(RNA_data)+'\n\nMET data:\n'+str(MET_data))
        self.is_processed = False
        
    def data_preprocessing(
        self,
        normalize_total=True,
        log1p=True,
        use_hvg=True,
        n_top_genes=3000,
        imputation = 'median',
        min_cells = 0.007,
        normalize='scale',
        add_noise = False,
        noise_rate = 0.0,
        noise_seed = 42,
        save_data=False,
        file_path=None,
        logging_path=None
    ):
        """
        Preprocessing for RNA data and methylation data in Bond.

        Parameters
        ----------
        normalize_total: bool
            choose use normalization or not, default True.

        log1p: bool
            choose use log transformation or not, default True.

        use_hvg: bool
            choose use highly variable genes or not, default True.

        n_top_genes: int
            the count of highly variable genes, if not use highly variable, set use_hvg = False and n_top_genes = None, default 3000.
            
        binary_data: bool
            choose binarized MET data or not, default True.

        filter_features: bool
            choose use peaks filtering or not, default True.

        fpeaks: float
            filter out the peaks expressed less than fpeaks*n_cells, if don't filter peaks set it None, default 0.005.

        tfidf: bool
            choose using TF-IDF transform or not, default True.

        normalize: bool
            choose scale data to [0, 1] or not, default True.

        save_data: bool
            choose save the processed data or not, default False.

        file_path: str
            the path for saving processed data, only used if save_data is True, default None.

        logging_path: str
            the path for output process logging, if not save, set it None, default None.
            
         """
        if self.is_processed:
            self.my_logger.warning('finding data have already been processed!')
        else:
            self.RNA_data_p = RNA_data_preprocessing(
                self.RNA_data,
                normalize_total=normalize_total,
                log1p=log1p,
                use_hvg=use_hvg,
                n_top_genes=n_top_genes,
                save_data=save_data,
                file_path=file_path,
                logging_path=logging_path
                )
            self.MET_data_p = MET_data_preprocessing(
                self.MET_data,
                imputation = imputation, 
                min_cells = min_cells,
                normalize=normalize,
                add_noise = add_noise,
                noise_rate = noise_rate,
                noise_seed = noise_seed,
                save_data=save_data,
                file_path=file_path,
                logging_path=logging_path
            )
            self.is_processed = True
            
    def augmentation(
        self,
        enable_augmentation=False
    ):
        """
        Data augmentation for Bond model (cell type only).
        
        Parameters
        ----------
        enable_augmentation: bool
            Whether to enable cell type-based data augmentation. Requires "cell_type" in RNA_data.obs.
            Default False (no augmentation).
        """
        if not enable_augmentation:
            self.my_logger.info('Data augmentation disabled.')
            return
        
        if 'cell_type' not in self.RNA_data.obs.keys():
            self.my_logger.warning('Skipping augmentation: "cell_type" not found in data.obs')
            return
        
        self.my_logger.info('using data augmentation with cell type labels.')
        copy_count = 3
        random.seed(19193)

        self.MET_data.obs.index = [str(i) for i in range(len(self.MET_data.obs.index))]
    
        cell_type = self.MET_data.obs.cell_type.iloc[self.train_id_m]
    
        for i in range(len(cell_type.cat.categories)):
            cell_type_name = cell_type.cat.categories[i]
            idx_temp = list(cell_type[cell_type == cell_type_name].index.astype(int))
        
            for j in range(copy_count - 1):
                random.shuffle(idx_temp)
                self.train_id_r.extend(idx_temp)
            
                random.shuffle(idx_temp)
                self.train_id_m.extend(idx_temp)
            
                        
    def construct_model(
        self,
        chrom_list,
        logging_path = None,
        R_encoder_nlayer = 2, 
        M_encoder_nlayer = 2,
        R_decoder_nlayer = 2, 
        M_decoder_nlayer = 2,
        R_encoder_dim_list = [256, 128],
        M_encoder_dim_list = [32, 128],
        R_decoder_dim_list = [128, 256],
        M_decoder_dim_list = [128, 32],
        R_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        M_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        M_decoder_act_list = [nn.LeakyReLU(), nn.Sigmoid()],
        translator_embed_dim = 128, 
        translator_input_dim_r = 128,
        translator_input_dim_m = 128,
        translator_embed_act_list = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        discriminator_nlayer = 1,
        discriminator_dim_list_R = [128],
        discriminator_dim_list_M = [128],
        discriminator_act_list = [nn.Sigmoid()],
        dropout_rate = 0.1,
        R_noise_rate = 0.5,
        M_noise_rate = 0.3,
        num_experts = 6,
        num_experts_single = 6,
        num_heads: int = 8,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1
    ):
        """
        Main model.
        
        Parameters
        ----------
        chrom_list: list
            list of peaks count for each chromosomes.
            
        logging_path: str
            the path for output process logging, if not save, set it None, default None.

        R_encoder_nlayer: int
            layer counts of RNA encoder, default 2.
            
        M_encoder_nlayer: int
            layer counts of methylation data encoder, default 2.
            
        R_decoder_nlayer: int
            layer counts of RNA decoder, default 2.
            
        M_decoder_nlayer: int
            layer counts of methylation data decoder, default 2.
            
        R_encoder_dim_list: list
            dimension list of RNA encoder, length equal to R_encoder_nlayer, default [256, 128].
            
        M_encoder_dim_list: list
            dimension list of methylation data encoder, length equal to M_encoder_nlayer, default [32, 128].
            
        R_decoder_dim_list: list
            dimension list of RNA decoder, length equal to R_decoder_nlayer, default [128, 256].
            
        M_decoder_dim_list: list
            dimension list of methylation data decoder, length equal to M_decoder_nlayer, default [128, 32].
            
        R_encoder_act_list: list
            activation list of RNA encoder, length equal to R_encoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].
            
        M_encoder_act_list: list
            activation list of methylation data encoder, length equal to M_encoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].
            
        R_decoder_act_list: list
            activation list of RNA decoder, length equal to R_decoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].
            
        M_decoder_act_list: list
            activation list of methylation data decoder, length equal to M_decoder_nlayer, default [nn.LeakyReLU(), nn.Sigmoid()].
            
        translator_embed_dim: int
            dimension of embedding space for translator, default 128.
            
        translator_input_dim_r: int
            dimension of input from RNA encoder for translator, default 128.
            
        translator_input_dim_a: int
            dimension of input from methylation data encoder for translator, default 128.
            
        translator_embed_act_list: list
            activation list for translator, involving [mean_activation, log_var_activation, decoder_activation], default [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()].
            
        discriminator_nlayer: int
            layer counts of discriminator, default 1.
            
        discriminator_dim_list_R: list
            dimension list of discriminator, length equal to discriminator_nlayer, the first equal to translator_input_dim_R, default [128].
            
        discriminator_dim_list_M: list
            dimension list of discriminator, length equal to discriminator_nlayer, the first equal to translator_input_dim_M, default [128].
            
        discriminator_act_list: list
            activation list of discriminator, length equal to  discriminator_nlayer, default [nn.Sigmoid()].
            
        dropout_rate: float
            rate of dropout for network, default 0.1.
       
        R_noise_rate: float
            rate of set part of RNA input data to 0, default 0.5.
            
        M_noise_rate: float
            rate of set part of methylation data input data to 0, default 0.3.
            
        num_experts: int
            number of experts for translator, default 6.
            
        num_experts: int
            number of experts for single translator, default 6.
            
        num_heads: int
            Number of parallel attention heads, default 8.
        
        attn_drop: float
            Dropout probability applied to attention weights, default 0.0.
        
        proj_drop: float
            Dropout probability applied to the output projection, default 0.0.

        """
        R_encoder_dim_list.insert(0, self.RNA_data_p.X.shape[1]) 
        M_encoder_dim_list.insert(0, self.MET_data_p.X.shape[1])
        R_decoder_dim_list.append(self.RNA_data_p.X.shape[1]) 
        M_decoder_dim_list.append(self.MET_data_p.X.shape[1])
        M_encoder_dim_list[1] *= len(chrom_list)
        M_decoder_dim_list[1] *= len(chrom_list)
        self.model = Model(
            R_encoder_nlayer = R_encoder_nlayer, 
            M_encoder_nlayer = M_encoder_nlayer,
            R_decoder_nlayer = R_decoder_nlayer, 
            M_decoder_nlayer = M_decoder_nlayer,
            R_encoder_dim_list = R_encoder_dim_list,
            M_encoder_dim_list = M_encoder_dim_list,
            R_decoder_dim_list = R_decoder_dim_list,
            M_decoder_dim_list = M_decoder_dim_list,
            R_encoder_act_list = R_encoder_act_list,
            M_encoder_act_list = M_encoder_act_list,
            R_decoder_act_list = R_decoder_act_list,
            M_decoder_act_list = M_decoder_act_list,
            translator_embed_dim = translator_embed_dim, 
            translator_input_dim_r = translator_input_dim_r,
            translator_input_dim_m = translator_input_dim_m,
            translator_embed_act_list = translator_embed_act_list,
            discriminator_nlayer = discriminator_nlayer,
            discriminator_dim_list_R = discriminator_dim_list_R,
            discriminator_dim_list_M = discriminator_dim_list_M,
            discriminator_act_list = discriminator_act_list,
            dropout_rate = dropout_rate,
            R_noise_rate = R_noise_rate,
            M_noise_rate = M_noise_rate,
            chrom_list = chrom_list,
            logging_path = logging_path,
            num_experts = num_experts,
            num_experts_single = num_experts_single,
            num_heads = num_heads,
            attn_drop = attn_drop,
            proj_drop = proj_drop,
            RNA_data = self.RNA_data_p,
            MET_data = self.MET_data_p
        )
        self.my_logger.info('successfully construct bond model.')

    def train_model(
        self,
        R_encoder_lr = 0.001,
        M_encoder_lr = 0.001,
        R_decoder_lr = 0.001,
        M_decoder_lr = 0.001,
        R_translator_lr = 0.0001,
        M_translator_lr = 0.0001,
        translator_lr = 0.0001,
        discriminator_lr = 0.005,
        R2R_pretrain_epoch = 100,
        M2M_pretrain_epoch = 100,
        lock_encoder_and_decoder = False,
        translator_epoch = 200,
        patience = 50,
        batch_size = 64,
        r_loss = nn.MSELoss(size_average=True),
        m_loss = nn.BCELoss(size_average=True),
        d_loss = nn.BCELoss(size_average=True),
        loss_weight = [1, 2, 1],
        output_path = None,
        seed = 19193,
        kl_mean = True,
        R_pretrain_kl_warmup = 50,
        M_pretrain_kl_warmup = 50,
        translation_kl_warmup = 50,
        load_model = None,
        logging_path = None
    ):
        """
        Training for model.
        
        Parameters
        ----------
        R_encoder_lr: float
            learning rate of RNA encoder, default 0.001.
            
        M_encoder_lr: float
            learning rate of methylation data encoder, default 0.001.
            
        R_decoder_lr: float
            learning rate of RNA decoder, default 0.001.
            
        M_decoder_lr: float
            learning rate of methylation data decoder, default 0.001.
       
        R_translator_lr: float
            learning rate of RNA pretrain translator, default 0.0001.
            
        M_translator_lr: float
            learning rate of methylation data pretrain translator, default 0.0001.
            
        translator_lr: float
            learning rate of translator, default 0.001.
            
        discriminator_lr: float
            learning rate of discriminator, default 0.005.
            
        R2R_pretrain_epoch: int
            max epoch for pretrain RNA autoencoder, default 100.
            
        M2M_pretrain_epoch: int
            max epoch for pretrain methylation data autoencoder, default 100.
            
        lock_encoder_and_decoder: bool
            lock the pretrained encoder and decoder or not, default False.
            
        translator_epoch: int
            max epoch for train translator, default 200.
            
        patience: int
            patience for loss on validation, default 50.
            
        batch_size: int
            batch size for training and validation, default 64.
            
        r_loss
            loss function for RNA reconstruction, default nn.MSELoss(size_average=True).
            
        m_loss
            loss function for methylation data reconstruction, default nn.BCELoss(size_average=True).
            
        d_loss
            loss function for discriminator, default nn.BCELoss(size_average=True).
            
        loss_weight: list
            list of loss weight for [r_loss, a_loss, d_loss], default [1, 2, 1].

        output_path: str
            file path for model output, default None.
            
        seed: int
            set up the random seed, default 19193.
            
        kl_mean: bool
            size average for kl divergence or not, default True.
            
        R_pretrain_kl_warmup: int
            epoch of linear weight warm up for kl divergence in RNA pretrain, default 50.
        
        M_pretrain_kl_warmup: int
            epoch of linear weight warm up for kl divergence in methylation data pretrain, default 50.
        
        translation_kl_warmup: int
            epoch of linear weight warm up for kl divergence in translator pretrain, default 50.
            
        load_model: str
            the path for loading model if needed, else set it None, default None.
            
        logging_path: str
            the path for output process logging, if not save, set it None, default None.
            
        """
        self.my_logger.info('training bond model ...')
        R_kl_div = 1 / self.RNA_data_p.X.shape[1] * 20
        M_kl_div = 1 / self.MET_data_p.X.shape[1] * 20
        kl_div = R_kl_div + M_kl_div
        loss_weight.extend([R_kl_div, M_kl_div, kl_div])
        self.model.train(
            R_encoder_lr = R_encoder_lr,
            M_encoder_lr = M_encoder_lr,
            R_decoder_lr = R_decoder_lr,
            M_decoder_lr = M_decoder_lr,
            R_translator_lr = R_translator_lr,
            M_translator_lr = M_translator_lr,
            translator_lr = translator_lr,
            discriminator_lr = discriminator_lr,
            R2R_pretrain_epoch = R2R_pretrain_epoch,
            M2M_pretrain_epoch = M2M_pretrain_epoch,
            lock_encoder_and_decoder = lock_encoder_and_decoder,
            translator_epoch = translator_epoch,
            patience = patience,
            batch_size = batch_size,
            r_loss = r_loss,
            m_loss = m_loss,
            d_loss = d_loss,
            loss_weight = loss_weight,
            train_id_r = self.train_id_r,
            train_id_m = self.train_id_m,
            validation_id_r = self.validation_id_r, 
            validation_id_m = self.validation_id_m, 
            output_path = output_path,
            seed = seed,
            kl_mean = kl_mean,
            R_pretrain_kl_warmup = R_pretrain_kl_warmup,
            M_pretrain_kl_warmup = M_pretrain_kl_warmup,
            translation_kl_warmup = translation_kl_warmup,
            load_model = load_model,
            logging_path = logging_path
        )

    def test_model(
        self, 
        model_path = None,
        load_model = False,
        output_path = None,
        test_cluster = False,
        test_figure = False,
        output_data = False
    ):
        """
        Test for model.
        
        Parameters
        ----------
        model_path: str
            path for load trained model, default None.
            
        load_model: bool
            load the pretrained model or not, default False.
            
        output_path: str
            file path for model output, default None.
            
        test_cluster: bool
            test clustrer index or not, default False.
            
        test_figure: bool
            test tSNE or not, default False.
            
        output_data: bool
            output the predicted test data to file or not, default False.
            
        """
        self.my_logger.info('testing bond model ...')
        M2R_predict, R2M_predict = self.model.test(
            test_id_r = self.test_id_r,
            test_id_m = self.test_id_m, 
            model_path = model_path,
            load_model = load_model,
            output_path = output_path,
            test_cluster = test_cluster,
            test_figure = test_figure,
            output_data = output_data,
            return_predict = True
        )
        return M2R_predict, R2M_predict

    def predict_single_rna(
        self,
        test_id_r=None,
        model_path=None,
        load_model=False,
        output_path=None,
        batch_size=64,
        return_embeddings=False
    ):
        """
        Predict methylation data from RNA data using existing test indices.
        
        Parameters
        ----------
        test_id_r: list
            List of RNA test indices. If None, will use test_id_r from loaded data.
            
        model_path: str
            Path for loading trained model, default None.
            
        load_model: bool
            Load the pretrained model or not, default True.
            
        output_path: str
            File path for model output, default None.
            
        batch_size: int
            Batch size for prediction, default 64.
            
        return_embeddings: bool
            Whether to return latent embeddings, default False.
            
        Returns
        -------
        AnnData
            Predicted methylation data.
        """
        # Use test_id_r from loaded data if not provided
        if test_id_r is None:
            test_id_r = self.test_id_r
        
        return self.model.predict_single_rna(
            test_id_r=test_id_r,
            model_path=model_path,
            load_model=load_model,
            output_path=output_path,
            batch_size=batch_size,
            return_embeddings=return_embeddings
        )

    def predict_single_methylation(
        self,
        test_id_m=None,
        model_path=None,
        load_model=False,
        output_path=None,
        batch_size=64,
        return_embeddings=False
    ):
        """
        Predict RNA data from methylation data using existing test indices.
        
        Parameters
        ----------
        test_id_m: list
            List of methylation test indices. If None, will use test_id_m from loaded data.
            
        model_path: str
            Path for loading trained model, default None.
            
        load_model: bool
            Load the pretrained model or not, default True.
            
        output_path: str
            File path for model output, default None.
            
        batch_size: int
            Batch size for prediction, default 64.
            
        return_embeddings: bool
            Whether to return latent embeddings, default False.
            
        Returns
        -------
        AnnData
            Predicted RNA data.
        """
        # Use test_id_m from loaded data if not provided
        if test_id_m is None:
            test_id_m = self.test_id_m
        
        return self.model.predict_single_methylation(
            test_id_m=test_id_m,
            model_path=model_path,
            load_model=load_model,
            output_path=output_path,
            batch_size=batch_size,
            return_embeddings=return_embeddings
        )

    def predict_single_modal(
        self,
        test_id=None,
        data_type='rna',
        model_path=None,
        load_model=False,
        output_path=None,
        batch_size=64,
        return_embeddings=False
    ):
        """
        Single modal prediction for RNA or methylation data using existing test indices.
        
        Parameters
        ----------
        test_id: list
            List of test indices. If None, will use test_id_r or test_id_m from loaded data.
            
        data_type: str
            Type of input data, 'rna' or 'methylation'/'met'/'dnam', default 'rna'.
            
        model_path: str
            Path for loading trained model, default None.
            
        load_model: bool
            Load the pretrained model or not, default True.
            
        output_path: str
            File path for model output, default None.
            
        batch_size: int
            Batch size for prediction, default 64.
            
        return_embeddings: bool
            Whether to return latent embeddings, default False.
            
        Returns
        -------
        AnnData or tuple
            Predicted data. If return_embeddings=True, returns (predicted_data, embeddings).
        """
        if data_type.lower() == 'rna':
            return self.predict_single_rna(
                test_id_r=test_id,
                model_path=model_path,
                load_model=load_model,
                output_path=output_path,
                batch_size=batch_size,
                return_embeddings=return_embeddings
            )
        elif data_type.lower() in ['methylation', 'met', 'dnam']:
            return self.predict_single_methylation(
                test_id_m=test_id,
                model_path=model_path,
                load_model=load_model,
                output_path=output_path,
                batch_size=batch_size,
                return_embeddings=return_embeddings
            )
        else:
            raise ValueError("data_type must be 'rna' or 'methylation'/'met'/'dnam'")
