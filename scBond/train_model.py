import os
import random
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scBridge.model_component import *
from scBridge.model_utlis import *
from scBridge.calculate_cluster import *
from scBridge.draw_cluster import *
from scBridge.data_processing import *
from scBridge.logger import *


class Model():
    def __init__(self,        
        RNA_data,
        MET_data,        
        chrom_list: list,
        logging_path: str,
        R_encoder_dim_list: list,
        M_encoder_dim_list: list,
        R_decoder_dim_list: list,
        M_decoder_dim_list: list,
        R_encoder_nlayer: int = 2, 
        M_encoder_nlayer: int = 2,
        R_decoder_nlayer: int = 2, 
        M_decoder_nlayer: int = 2,
        R_encoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        M_encoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        M_decoder_act_list: list = [nn.LeakyReLU(), nn.Sigmoid()],
        translator_embed_dim: int = 128, 
        translator_input_dim_r: int = 128,
        translator_input_dim_m: int = 128,
        translator_embed_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],         
        discriminator_nlayer: int = 1,
        discriminator_dim_list_R: list = [128],
        discriminator_dim_list_M: list = [128],
        discriminator_act_list: list = [nn.Sigmoid()],
        dropout_rate: float = 0.1,
        R_noise_rate: float = 0.5,
        M_noise_rate: float = 0.3,
        num_experts: int = 6, 
        num_experts_single: int = 6,
        reduction_ratio: int = 32,  
        num_heads: int = 8,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1
    ):
        """
        Main model. Some parameters need information about data, please see in Tutorial.
        
        Parameters
        ----------
        RNA_data: Anndata
            RNA data for model training and testing.
            
        MET_data: Anndata
            Methylation data for model training and testing.

        chrom_list: list
            list of peaks count for each chromosomes.
            
        logging_path: str
            the path for output process logging, if not save, set it None.
            
        R_encoder_dim_list: list
            dimension list of RNA encoder, length equal to R_encoder_nlayer + 1, the first equal to RNA data dimension, the last equal to embedding dimension.
            
        M_encoder_dim_list: list
            dimension list of methylation data encoder, length equal to A_encoder_nlayer + 1, the first equal to RNA data dimension, the last equal to embedding dimension.
            
        R_decoder_dim_list: list
            dimension list of RNA decoder, length equal to R_decoder_nlayer + 1, the last equal to embedding dimension, the first equal to RNA data dimension.
            
        M_decoder_dim_list: list
            dimension list of methylation data decoder, length equal to A_decoder_nlayer + 1, the last equal to embedding dimension, the first equal to RNA data dimension.
            
        R_encoder_nlayer: int
            layer counts of RNA encoder, default 2.
            
        M_encoder_nlayer: int
            layer counts of methylation data encoder, default 2.
            
        R_decoder_nlayer: int
            layer counts of RNA decoder, default 2.
            
        M_decoder_nlayer: int
            layer counts of methylation data decoder, default 2.
            
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
            
        translator_input_dim_m: int
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

        """
        if not logging_path is None:
            file_handle=open(logging_path + '/Parameters_Record.txt',mode='a')
            file_handle.writelines([
                '------------------------------\n'
                'Model Parameters\n'
                'R_encoder_nlayer: '+str(R_encoder_nlayer)+'\n', 
                'M_encoder_nlayer: '+str(M_encoder_nlayer)+'\n',
                'R_decoder_nlayer: '+str(R_decoder_nlayer)+'\n', 
                'M_decoder_nlayer: '+str(M_decoder_nlayer)+'\n',
                'R_encoder_dim_list: '+str(R_encoder_dim_list)+'\n',
                'M_encoder_dim_list: '+str(M_encoder_dim_list)+'\n',
                'R_decoder_dim_list: '+str(R_decoder_dim_list)+'\n',
                'M_decoder_dim_list: '+str(M_decoder_dim_list)+'\n',
                'R_encoder_act_list: '+str(R_encoder_act_list)+'\n',
                'M_encoder_act_list: '+str(M_encoder_act_list)+'\n',
                'R_decoder_act_list: '+str(R_decoder_act_list)+'\n',
                'M_decoder_act_list: '+str(M_decoder_act_list)+'\n',
                'translator_embed_dim: '+str(translator_embed_dim)+'\n', 
                'translator_input_dim_r: '+str(translator_input_dim_r)+'\n',
                'translator_input_dim_a: '+str(translator_input_dim_m)+'\n',
                'translator_embed_act_list: '+str(translator_embed_act_list)+'\n',
                'discriminator_nlayer: '+str(discriminator_nlayer)+'\n',
                'discriminator_dim_list_R: '+str(discriminator_dim_list_R)+'\n',
                'discriminator_dim_list_M: '+str(discriminator_dim_list_M)+'\n',
                'discriminator_act_list: '+str(discriminator_act_list)+'\n',
                'dropout_rate: '+str(dropout_rate)+'\n',
                'R_noise_rate: '+str(R_noise_rate)+'\n',
                'M_noise_rate: '+str(M_noise_rate)+'\n',
                'num_experts: '+str(num_experts)+'\n',
                'num_experts_single: '+str(num_experts_single)+'\n',
                'chrom_list: '+str(chrom_list)+'\n'
                'num_heads: '+str(num_heads)+'\n',
                'attn_drop: '+str(attn_drop)+'\n',
                'proj_drop: '+str(proj_drop)+'\n'
            ])
            file_handle.close()
        
        self.RNA_encoder = NetBlock(
            nlayer = R_encoder_nlayer,
            dim_list = R_encoder_dim_list,
            act_list = R_encoder_act_list,
            dropout_rate = dropout_rate,
            noise_rate = R_noise_rate)

        self.MET_encoder = Split_Chrom_Encoder_block(
            nlayer = M_encoder_nlayer,
            dim_list = M_encoder_dim_list,
            act_list = M_encoder_act_list,
            chrom_list = chrom_list,
            dropout_rate = dropout_rate,
            noise_rate = M_noise_rate)
            
        self.RNA_decoder = NetBlock(
            nlayer = R_decoder_nlayer,
            dim_list = R_decoder_dim_list,
            act_list = R_decoder_act_list,
            dropout_rate = dropout_rate,
            noise_rate = 0)
            
        self.MET_decoder = Split_Chrom_Decoder_block(
            nlayer = M_decoder_nlayer,
            dim_list = M_decoder_dim_list,
            act_list = M_decoder_act_list,
            chrom_list = chrom_list,
            dropout_rate = dropout_rate,
            noise_rate = 0)
        
        self.R_translator = Single_Translator(
            translator_input_dim = translator_input_dim_r, 
            translator_embed_dim = translator_embed_dim, 
            translator_embed_act_list = translator_embed_act_list,
            num_experts_single = num_experts_single,
            reduction_ratio = reduction_ratio,
            num_heads = num_heads,
            attn_drop = attn_drop,
            proj_drop = proj_drop
        )
        
        self.M_translator = Single_Translator(
            translator_input_dim = translator_input_dim_m, 
            translator_embed_dim = translator_embed_dim, 
            translator_embed_act_list = translator_embed_act_list,
            num_experts_single = num_experts_single,
            reduction_ratio = reduction_ratio,
            num_heads = num_heads,
            attn_drop = attn_drop,
            proj_drop = proj_drop
        )
        
        self.translator = Translator(
            translator_input_dim_r = translator_input_dim_r,
            translator_input_dim_m = translator_input_dim_m,
            translator_embed_dim = translator_embed_dim,
            translator_embed_act_list = translator_embed_act_list,
            num_experts = num_experts,
            reduction_ratio = reduction_ratio,
            num_heads = num_heads,
            attn_drop = attn_drop,
            proj_drop = proj_drop
        )

        discriminator_dim_list_R.append(1)
        discriminator_dim_list_M.append(1)
        self.discriminator_R = NetBlock(
            nlayer = discriminator_nlayer,
            dim_list = discriminator_dim_list_R,
            act_list = discriminator_act_list,
            dropout_rate = 0,
            noise_rate = 0)

        self.discriminator_M = NetBlock(
            nlayer = discriminator_nlayer,
            dim_list = discriminator_dim_list_M,
            act_list = discriminator_act_list,
            dropout_rate = 0,
            noise_rate = 0)
        
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                self.RNA_encoder = self.RNA_encoder.cuda()
                self.RNA_decoder = self.RNA_decoder.cuda()
                self.MET_encoder = self.MET_encoder.cuda()
                self.MET_decoder = self.MET_decoder.cuda()
                self.R_translator = self.R_translator.cuda()
                self.M_translator = self.M_translator.cuda()
                self.translator = self.translator.cuda()
                self.discriminator_R = self.discriminator_R.cuda()
                self.discriminator_M = self.discriminator_M.cuda()
        
        self.is_train_finished = False
        self.RNA_data_obs = RNA_data.obs
        self.MET_data_obs = MET_data.obs
        self.RNA_data_var = RNA_data.var
        self.MET_data_var = MET_data.var
        if isinstance(RNA_data.X, np.ndarray):
            self.RNA_data = RNA_data.X
        else:
            self.RNA_data = RNA_data.X.toarray()
        if isinstance(MET_data.X, np.ndarray):
            self.MET_data = MET_data.X
        else:
            self.MET_data = MET_data.X.toarray()
    
    def set_train(self):
        self.RNA_encoder.train()
        self.RNA_decoder.train()
        self.MET_encoder.train()
        self.MET_decoder.train()
        self.R_translator.train()
        self.M_translator.train()
        self.translator.train()
        self.discriminator_R.train()
        self.discriminator_M.train()
     
    
    def set_eval(self):
        self.RNA_encoder.eval()
        self.RNA_decoder.eval()
        self.MET_encoder.eval()
        self.MET_decoder.eval()
        self.R_translator.eval()
        self.M_translator.eval()        
        self.translator.eval()
        self.discriminator_R.eval()
        self.discriminator_M.eval()
    
        
    def forward_R2R(self, RNA_input, r_loss, kl_div_w, forward_type):
        latent_layer, mu, d = self.R_translator(self.RNA_encoder(RNA_input), forward_type)
        predict_RNA = self.RNA_decoder(latent_layer)
        reconstruct_loss = r_loss(predict_RNA, RNA_input)
        kl_div_r = - 0.5 * torch.mean(1 + d - mu.pow(2) - d.exp())
        loss = reconstruct_loss + kl_div_w * kl_div_r
        return loss, reconstruct_loss, kl_div_r
    
    
    def forward_M2M(self, MET_input, m_loss, kl_div_w, forward_type):
        latent_layer, mu, d = self.M_translator(self.MET_encoder(MET_input), forward_type)
        predict_MET = self.MET_decoder(latent_layer)
        reconstruct_loss = m_loss(predict_MET, MET_input)
        kl_div_m = - 0.5 * torch.mean(1 + d - mu.pow(2) - d.exp())
        loss = reconstruct_loss + kl_div_w * kl_div_m
        return loss, reconstruct_loss, kl_div_m
        
       
    def forward_translator(self, batch_samples, RNA_input_dim, MET_input_dim, m_loss, r_loss, loss_weight, forward_type, kl_div_mean=False):
    
        RNA_input, MET_input = torch.split(batch_samples, [RNA_input_dim, MET_input_dim], dim=1)
        
        R2 = self.RNA_encoder(RNA_input)
        M2 = self.MET_encoder(MET_input)
        if forward_type == 'train':
            R2R, R2M, mu_r, sigma_r = self.translator.train_model(R2, 'RNA')
            M2R, M2M, mu_m, sigma_m = self.translator.train_model(M2, 'MET')
        elif forward_type == 'test':
            R2R, R2M, mu_r, sigma_r = self.translator.test_model(R2, 'RNA')
            M2R, M2M, mu_m, sigma_m = self.translator.test_model(M2, 'MET')        
        
        R2R = self.RNA_decoder(R2R)
        R2M = self.MET_decoder(R2M)
        M2R = self.RNA_decoder(M2R)
        M2M = self.MET_decoder(M2M)
        
        # reconstruct loss
        lossR2R = r_loss(R2R, RNA_input)
        lossM2R = r_loss(M2R, RNA_input)
        lossR2M = m_loss(R2M, MET_input)
        lossM2M = m_loss(M2M, MET_input)
        
        # kl divergence
        if kl_div_mean:
            kl_div_r = - 0.5 * torch.mean(1 + sigma_r - mu_r.pow(2) - sigma_r.exp())
            kl_div_m = - 0.5 * torch.mean(1 + sigma_m - mu_m.pow(2) - sigma_m.exp())
        else:
            kl_div_r = torch.clamp(- 0.5 * torch.sum(1 + sigma_r - mu_r.pow(2) - sigma_r.exp()), 0, 10000)
            kl_div_m = torch.clamp(- 0.5 * torch.sum(1 + sigma_m - mu_m.pow(2) - sigma_m.exp()), 0, 10000)
        
        # calculate the loss
        r_loss_w, m_loss_w, d_loss_w, kl_div_R, kl_div_M, kl_div_w = loss_weight
        reconstruct_loss = r_loss_w * (lossR2R + lossM2R) + m_loss_w * (lossR2M + lossM2M)

        kl_div = kl_div_r + kl_div_m
        
        loss_g = kl_div_w * kl_div + reconstruct_loss

        return reconstruct_loss, kl_div, loss_g
    
    
    def forward_discriminator(self, batch_samples, RNA_input_dim, MET_input_dim, d_loss, forward_type):
        
        RNA_input, MET_input = torch.split(batch_samples, [RNA_input_dim, MET_input_dim], dim=1)

        # forward of generator
        R2 = self.RNA_encoder(RNA_input)
        M2 = self.MET_encoder(MET_input)
        if forward_type == 'train':
            R2R, R2M, mu_r, sigma_r = self.translator.train_model(R2, 'RNA')
            M2R, M2M, mu_m, sigma_m = self.translator.train_model(M2, 'MET')
        elif forward_type == 'test':
            R2R, R2M, mu_r, sigma_r = self.translator.test_model(R2, 'RNA')
            M2R, M2M, mu_m, sigma_m = self.translator.test_model(M2, 'MET')

        batch_size = batch_samples.shape[0]

        temp1 = np.random.rand(batch_size)
        temp = [0 for item in temp1]
        for i in range(len(temp1)):
            if temp1[i] > 0.8:
                temp[i] = temp1[i]
            elif temp1[i] <= 0.8 and temp1[i] > 0.5:
                temp[i] = 0.8
            elif temp1[i] <= 0.5 and temp1[i] > 0.2:
                temp[i] = 0.2
            else:
                temp[i] = temp1[i]

        input_data_m = torch.stack([M2[i] if temp[i] > 0.5 else R2M[i] for i in range(batch_size)], dim=0)
        input_data_r = torch.stack([R2[i] if temp[i] > 0.5 else M2R[i] for i in range(batch_size)], dim=0)

        predict_met = self.discriminator_M(input_data_m)
        predict_rna = self.discriminator_R(input_data_r)

        loss1 = d_loss(predict_met.reshape(batch_size), torch.tensor(temp).cuda().float())
        loss2 = d_loss(predict_rna.reshape(batch_size), torch.tensor(temp).cuda().float())
        return loss1 + loss2
        
    
    def save_model_dict(self, output_path):
        torch.save(self.RNA_encoder.state_dict(), output_path + '/model/RNA_encoder.pt')
        torch.save(self.MET_encoder.state_dict(), output_path + '/model/MET_encoder.pt')
        torch.save(self.RNA_decoder.state_dict(), output_path + '/model/RNA_decoder.pt')
        torch.save(self.MET_decoder.state_dict(), output_path + '/model/MET_decoder.pt')
        torch.save(self.R_translator.state_dict(), output_path + '/model/R_translator.pt')
        torch.save(self.M_translator.state_dict(), output_path + '/model/M_translator.pt')
        torch.save(self.translator.state_dict(), output_path + '/model/translator.pt')
        torch.save(self.discriminator_M.state_dict(), output_path + '/model/discriminator_M.pt')
        torch.save(self.discriminator_R.state_dict(), output_path + '/model/discriminator_R.pt')


    def train(
        self,
        loss_weight: list,
        train_id_r: list,
        train_id_m: list,
        validation_id_r: list,
        validation_id_m: list,
        R_encoder_lr: float = 0.001,
        M_encoder_lr: float = 0.001,
        R_decoder_lr: float = 0.001,
        M_decoder_lr: float = 0.001,
        R_translator_lr: float = 0.0001,
        M_translator_lr: float = 0.0001,
        translator_lr: float = 0.0001,
        discriminator_lr: float = 0.001,
        R2R_pretrain_epoch: int = 100,
        M2M_pretrain_epoch: int = 100,
        lock_encoder_and_decoder: bool = False,
        translator_epoch: int = 200,
        patience: int = 50,
        batch_size: int = 64,
        r_loss = nn.MSELoss(size_average=True),
        m_loss = nn.BCELoss(size_average=True),
        d_loss = nn.BCELoss(size_average=True),
        output_path: str = None,
        seed: int = 19193,
        kl_mean: bool = True,
        R_pretrain_kl_warmup: int = 50,
        M_pretrain_kl_warmup: int = 50,
        translation_kl_warmup: int = 50,
        load_model: str = None,
        logging_path: str = None
    ):
        """
        Training for model. Some parameters need information about data, please see in Tutorial.
        
        Parameters
        ----------
        loss_weight: list
            list of loss weight for [r_loss, m_loss, d_loss, kl_div_R, kl_div_M, kl_div_all].
        
        train_id_r: list
            list of RNA data cell ids for training.
            
        train_id_m: list
            list of methylation data cell ids for training.
            
        validation_id_r: list
            list of RNA data cell ids for validation.
        
        validation_id_a: list
            list of methylation data cell ids for validation.

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
            learning rate of translator, default 0.0001.
            
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
        if not logging_path is None:
            file_handle=open(logging_path + '/Parameters_Record.txt',mode='a')
            file_handle.writelines([
                '------------------------------\n'
                'Train Parameters\n'
                'R_encoder_lr: '+str(R_encoder_lr)+'\n',
                'A_encoder_lr: '+str(M_encoder_lr)+'\n',
                'R_decoder_lr: '+str(R_decoder_lr)+'\n',
                'A_decoder_lr: '+str(M_decoder_lr)+'\n',
                'R_translator_lr: '+str(R_translator_lr)+'\n',
                'A_translator_lr: '+str(M_translator_lr)+'\n',
                'translator_lr: '+str(translator_lr)+'\n',
                'discriminator_lr: '+str(discriminator_lr)+'\n',
                'R2R_pretrain_epoch: '+str(R2R_pretrain_epoch)+'\n',
                'A2A_pretrain_epoch: '+str(M2M_pretrain_epoch)+'\n',
                'lock_encoder_and_decoder: '+str(lock_encoder_and_decoder)+'\n',
                'translator_epoch: '+str(translator_epoch)+'\n',
                'patience: '+str(patience)+'\n',
                'batch_size: '+str(batch_size)+'\n',
                'r_loss: '+str(r_loss)+'\n',
                'a_loss: '+str(m_loss)+'\n',
                'd_loss: '+str(d_loss)+'\n',
                'loss_weight: '+str(loss_weight)+'\n',
                'seed: '+str(seed)+'\n',
                'kl_mean: '+str(kl_mean)+'\n',
                'R_pretrain_kl_warmup: '+str(R_pretrain_kl_warmup)+'\n',
                'A_pretrain_kl_warmup: '+str(M_pretrain_kl_warmup)+'\n',
                'translation_kl_warmup: '+str(translation_kl_warmup)+'\n',
                'load_model: '+str(load_model)+'\n'
            ])
            file_handle.close()
            
        my_logger = create_logger(name='Trainer', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
        
        self.is_train_finished = False
        
        if output_path is None:
            output_path = '.'
        
        if not load_model is None:
            my_logger.info('load pretrained model from path: '+str(load_model)+'/model/')
            self.RNA_encoder.load_state_dict(torch.load(load_model + '/model/RNA_encoder.pt'))
            self.MET_encoder.load_state_dict(torch.load(load_model + '/model/MET_encoder.pt'))
            self.RNA_decoder.load_state_dict(torch.load(load_model + '/model/RNA_decoder.pt'))
            self.MET_decoder.load_state_dict(torch.load(load_model + '/model/MET_decoder.pt'))
            self.translator.load_state_dict(torch.load(load_model + '/model/translator.pt'))
            self.discriminator_M.load_state_dict(torch.load(load_model + '/model/discriminator_M.pt'))
            self.discriminator_R.load_state_dict(torch.load(load_model + '/model/discriminator_R.pt'))
            
        if not seed is None:
            setup_seed(seed)
        
        RNA_input_dim = self.RNA_data.shape[1]
        MET_input_dim = self.MET_data.shape[1]
        cell_count = len(train_id_r)
        
        # define the dataset and dataloader for train and validation
        self.train_dataset = RNA_MET_dataset(self.RNA_data, self.MET_data, train_id_r, train_id_m)
        self.validation_dataset = RNA_MET_dataset(self.RNA_data, self.MET_data, validation_id_r, validation_id_m)

        if cell_count % batch_size == 1:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        else:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        self.optimizer_R_encoder = torch.optim.Adam(self.RNA_encoder.parameters(), lr=R_encoder_lr)
        self.optimizer_M_encoder = torch.optim.Adam(self.MET_encoder.parameters(), lr=M_encoder_lr, weight_decay=0)
        self.optimizer_R_decoder = torch.optim.Adam(self.RNA_decoder.parameters(), lr=R_decoder_lr)
        self.optimizer_M_decoder = torch.optim.Adam(self.MET_decoder.parameters(), lr=M_decoder_lr, weight_decay=0)
        self.optimizer_R_translator = torch.optim.Adam(self.R_translator.parameters(), lr=R_translator_lr)
        self.optimizer_M_translator = torch.optim.Adam(self.M_translator.parameters(), lr=M_translator_lr)
        self.optimizer_translator = torch.optim.Adam(self.translator.parameters(), lr=translator_lr)
        self.optimizer_discriminator_M = torch.optim.SGD(self.discriminator_M.parameters(), lr=discriminator_lr)
        self.optimizer_discriminator_R = torch.optim.SGD(self.discriminator_R.parameters(), lr=discriminator_lr)
        
        """ eraly stop for model """
        self.early_stopping_R2R = EarlyStopping(patience=patience, verbose=False)
        self.early_stopping_M2M = EarlyStopping(patience=patience, verbose=False)
        self.early_stopping_all = EarlyStopping(patience=patience, verbose=False)
        
        if not os.path.exists(output_path + '/model'):
            os.mkdir(output_path + '/model')
        
        """ pretrain for RNA and Methylation """
        
        pretrain_m_loss, pretrain_m_kl, pretrain_m_loss_val, pretrain_m_kl_val = [], [], [], []
        my_logger.info('MET pretraining ...')
        with tqdm(total = M2M_pretrain_epoch, ncols=100) as pbar:
            pbar.set_description('MET pretrain')
            for epoch in range(M2M_pretrain_epoch):
                pretrain_m_loss_, pretrain_m_kl_, pretrain_m_loss_val_, pretrain_m_kl_val_ = [], [], [], []
                self.set_train()
                for idx, batch_samples in enumerate(self.train_dataloader): 

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, MET_input = torch.split(batch_samples, [RNA_input_dim, MET_input_dim], dim=1)

                    """ pretrain for MET """
                    weight_temp = loss_weight.copy()
                    if epoch < M_pretrain_kl_warmup:
                        weight_temp[4] = loss_weight[4] * epoch / M_pretrain_kl_warmup

                    loss, reconstruct_loss, kl_div_m = self.forward_M2M(MET_input,m_loss, weight_temp[4], 'train')
                    self.optimizer_M_encoder.zero_grad()
                    self.optimizer_M_decoder.zero_grad()
                    self.optimizer_M_translator.zero_grad()
                    loss.backward()
                    self.optimizer_M_encoder.step()
                    self.optimizer_M_decoder.step()
                    self.optimizer_M_translator.step()

                    pretrain_m_loss_.append(reconstruct_loss.item())
                    pretrain_m_kl_.append(kl_div_m.item())

                self.set_eval()
                for idx, batch_samples in enumerate(self.validation_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, MET_input = torch.split(batch_samples, [RNA_input_dim, MET_input_dim], dim=1)

                    loss, reconstruct_loss, kl_div_m = self.forward_M2M(MET_input, m_loss, weight_temp[4], 'test')

                    pretrain_m_loss_val_.append(reconstruct_loss.item())
                    pretrain_m_kl_val_.append(kl_div_m.item())

                pretrain_m_loss.append(np.mean(pretrain_m_loss_))
                pretrain_m_kl.append(np.mean(pretrain_m_kl_))
                pretrain_m_loss_val.append(np.mean(pretrain_m_loss_val_))
                pretrain_m_kl_val.append(np.mean(pretrain_m_kl_val_))

                self.early_stopping_M2M(np.mean(pretrain_m_loss_val_), self, output_path)
                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train='{:.4f}'.format(np.mean(pretrain_m_loss_val_)), 
                    val='{:.4f}'.format(np.mean(pretrain_m_loss_)))
                
                if self.early_stopping_M2M.early_stop:
                    my_logger.info('MET pretraining early stop, validation loss does not improve in '+str(patience)+' epoches!')
                    self.MET_encoder.load_state_dict(torch.load(output_path + '/model/MET_encoder.pt'))
                    self.MET_decoder.load_state_dict(torch.load(output_path + '/model/MET_decoder.pt'))
                    self.M_translator.load_state_dict(torch.load(output_path + '/model/M_translator.pt'))
                    break
                    
                    
        my_logger.info('RNA pretraining ...')
        pretrain_r_loss, pretrain_r_kl, pretrain_r_loss_val, pretrain_r_kl_val = [], [], [], []
        with tqdm(total = R2R_pretrain_epoch, ncols=100) as pbar:
            pbar.set_description('RNA pretrain')
            for epoch in range(R2R_pretrain_epoch):
                pretrain_r_loss_, pretrain_r_kl_, pretrain_r_loss_val_, pretrain_r_kl_val_ = [], [], [], []
                self.set_train()
                for idx, batch_samples in enumerate(self.train_dataloader): 

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, MET_input = torch.split(batch_samples, [RNA_input_dim, MET_input_dim], dim=1)

                    """ pretrain for RNA """
                    weight_temp = loss_weight.copy()
                    if epoch < R_pretrain_kl_warmup:
                        weight_temp[3] = loss_weight[3] * epoch / R_pretrain_kl_warmup

                    loss, reconstruct_loss, kl_div_r = self.forward_R2R(RNA_input, r_loss, weight_temp[3], 'train')
                    self.optimizer_R_encoder.zero_grad()
                    self.optimizer_R_decoder.zero_grad()
                    self.optimizer_R_translator.zero_grad()
                    loss.backward()
                    self.optimizer_R_encoder.step()
                    self.optimizer_R_decoder.step()
                    self.optimizer_R_translator.step()

                    pretrain_r_loss_.append(reconstruct_loss.item())
                    pretrain_r_kl_.append(kl_div_r.item())

                self.set_eval()
                for idx, batch_samples in enumerate(self.validation_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, MET_input = torch.split(batch_samples, [RNA_input_dim, MET_input_dim], dim=1)

                    loss, reconstruct_loss, kl_div_r = self.forward_R2R(RNA_input, r_loss, weight_temp[3], 'test')

                    pretrain_r_loss_val_.append(reconstruct_loss.item())
                    pretrain_r_kl_val_.append(kl_div_r.item())

                pretrain_r_loss.append(np.mean(pretrain_r_loss_))
                pretrain_r_kl.append(np.mean(pretrain_r_kl_))
                pretrain_r_loss_val.append(np.mean(pretrain_r_loss_val_))
                pretrain_r_kl_val.append(np.mean(pretrain_r_kl_val_))

                self.early_stopping_R2R(np.mean(pretrain_r_loss_val_), self, output_path)
                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train='{:.4f}'.format(np.mean(pretrain_r_loss_val_)), 
                    val='{:.4f}'.format(np.mean(pretrain_r_loss_)))
                
                if self.early_stopping_R2R.early_stop:
                    my_logger.info('RNA pretraining early stop, validation loss does not improve in '+str(patience)+' epoches!')
                    self.RNA_encoder.load_state_dict(torch.load(output_path + '/model/RNA_encoder.pt'))
                    self.RNA_decoder.load_state_dict(torch.load(output_path + '/model/RNA_decoder.pt'))
                    self.R_translator.load_state_dict(torch.load(output_path + '/model/R_translator.pt'))
                    break

        
        """ train for translator and discriminator """
        train_loss, train_kl, train_discriminator, train_loss_val, train_kl_val, train_discriminator_val = [], [], [], [], [], []
        my_logger.info('Integrative training ...')
        with tqdm(total = translator_epoch, ncols=100) as pbar:
            pbar.set_description('Integrative training')
            for epoch in range(translator_epoch):
                train_loss_, train_kl_, train_discriminator_, train_loss_val_, train_kl_val_, train_discriminator_val_ = [], [], [], [], [], []
                self.set_train()
                for idx, batch_samples in enumerate(self.train_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, MET_input = torch.split(batch_samples, [RNA_input_dim, MET_input_dim], dim=1)

                    """ train for discriminator """
                    loss_d = self.forward_discriminator(batch_samples, RNA_input_dim, MET_input_dim, d_loss, 'train')
                    self.optimizer_discriminator_R.zero_grad()
                    self.optimizer_discriminator_M.zero_grad()
                    loss_d.backward()
                    self.optimizer_discriminator_R.step()
                    self.optimizer_discriminator_M.step()

                    """ train for generator """
                    weight_temp = loss_weight.copy()
                    if epoch < translation_kl_warmup:
                        weight_temp[5] = loss_weight[5] * epoch / translation_kl_warmup
                    loss_d = self.forward_discriminator(batch_samples, RNA_input_dim, MET_input_dim, d_loss, 'train')
                    reconstruct_loss, kl_div, loss_g = self.forward_translator(batch_samples, RNA_input_dim, MET_input_dim, m_loss, r_loss, weight_temp, 'train', kl_mean)

                    if loss_d.item() < 1.35:
                        loss_g -= loss_weight[2] * loss_d

                    self.optimizer_translator.zero_grad()
                    if not lock_encoder_and_decoder:
                        self.optimizer_R_encoder.zero_grad()
                        self.optimizer_M_encoder.zero_grad()
                        self.optimizer_R_decoder.zero_grad()
                        self.optimizer_M_decoder.zero_grad()
                    loss_g.backward()
                    self.optimizer_translator.step()
                    if not lock_encoder_and_decoder:
                        self.optimizer_R_encoder.step()
                        self.optimizer_M_encoder.step()
                        self.optimizer_R_decoder.step()
                        self.optimizer_M_decoder.step()

                    train_loss_.append(reconstruct_loss.item())
                    train_kl_.append(kl_div.item()) 
                    train_discriminator_.append(loss_d.item())

                self.set_eval()
                for idx, batch_samples in enumerate(self.validation_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)                    

                    RNA_input, MET_input = torch.split(batch_samples, [RNA_input_dim, MET_input_dim], dim=1)

                    """ test for discriminator """
                    loss_d = self.forward_discriminator(batch_samples, RNA_input_dim, MET_input_dim, d_loss, 'test')

                    """ test for generator """
                    loss_d = self.forward_discriminator(batch_samples, RNA_input_dim, MET_input_dim, d_loss, 'test')
                    reconstruct_loss, kl_div, loss_g = self.forward_translator(batch_samples, RNA_input_dim, MET_input_dim, m_loss, r_loss, weight_temp, 'train', kl_mean)
                    loss_g -= loss_weight[2] * loss_d

                    train_loss_val_.append(reconstruct_loss.item())
                    train_kl_val_.append(kl_div.item()) 
                    train_discriminator_val_.append(loss_d.item())

                train_loss.append(np.mean(train_loss_))
                train_kl.append(np.mean(train_kl_))
                train_discriminator.append(np.mean(train_discriminator_))
                train_loss_val.append(np.mean(train_loss_val_))
                train_kl_val.append(np.mean(train_kl_val_))
                train_discriminator_val.append(np.mean(train_discriminator_val_))
                self.early_stopping_all(np.mean(train_loss_val_), self, output_path)
                
                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train='{:.4f}'.format(np.mean(train_loss_val_)), 
                    val='{:.4f}'.format(np.mean(train_loss_)))
                
                if self.early_stopping_all.early_stop:
                    my_logger.info('Integrative training early stop, validation loss does not improve in '+str(patience)+' epoches!')
                    self.RNA_encoder.load_state_dict(torch.load(output_path + '/model/RNA_encoder.pt'))
                    self.MET_encoder.load_state_dict(torch.load(output_path + '/model/MET_encoder.pt'))
                    self.RNA_decoder.load_state_dict(torch.load(output_path + '/model/RNA_decoder.pt'))
                    self.MET_decoder.load_state_dict(torch.load(output_path + '/model/MET_decoder.pt'))
                    self.translator.load_state_dict(torch.load(output_path + '/model/translator.pt'))
                    self.discriminator_M.load_state_dict(torch.load(output_path + '/model/discriminator_M.pt'))
                    self.discriminator_R.load_state_dict(torch.load(output_path + '/model/discriminator_R.pt'))
                    break
        
        self.save_model_dict(output_path)
            
        self.is_train_finished = True
        
        record_loss_log(
            pretrain_r_loss,
            pretrain_r_kl,
            pretrain_r_loss_val,
            pretrain_r_kl_val,
            pretrain_m_loss,
            pretrain_m_kl,
            pretrain_m_loss_val,
            pretrain_m_kl_val,
            train_loss,
            train_kl,
            train_discriminator,
            train_loss_val,
            train_kl_val,
            train_discriminator_val,
            output_path
        )

    def test(
        self,
        test_id_r: list,
        test_id_m: list,
        model_path: str = None,
        load_model: bool = True,
        output_path: str = None,
        test_cluster: bool = True,
        test_figure: bool = True,
        output_data: bool = False,
        return_predict: bool = False
    ):
        """
        Test for model.
        
        Parameters
        ----------            
        train_id_r: list
            list of RNA data cell ids for training.
            
        train_id_m: list
            list of methylation data cell ids for training.
            
        model_path: str
            path for load trained model, default None.
            
        load_model: bool
            load the pretrained model or not, deafult True.
            
        output_path: str
            file path for model output, default None.
            
        test_cluster: bool
            test clustrer index or not, deafult True.
            
        test_figure: bool
            test tSNE or not, deafult True.
            
        output_data: bool
            output the predicted test data to file or not, deafult False.
            
        return_predict: bool
            return predict or not, if True, output (M2R_predict, R2M_predict) as returns, deafult False.
            
        """
        my_logger = create_logger(name='Tester', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
        
        if output_path is None:
            output_path = '.'
        
        """ load model from model_path if need """
        if load_model:
            my_logger.info('load trained model from path: '+str(model_path)+'/model')
            self.RNA_encoder.load_state_dict(torch.load(model_path + '/model/RNA_encoder.pt'))
            self.MET_encoder.load_state_dict(torch.load(model_path + '/model/MET_encoder.pt'))
            self.RNA_decoder.load_state_dict(torch.load(model_path + '/model/RNA_decoder.pt'))
            self.MET_decoder.load_state_dict(torch.load(model_path + '/model/MET_decoder.pt'))
            self.translator.load_state_dict(torch.load(model_path + '/model/translator.pt'))
        
        """ load data """
        RNA_input_dim = self.RNA_data.shape[1]
        MET_input_dim = self.MET_data.shape[1]
        
        self.R_test_dataset = Single_omics_dataset(self.RNA_data, test_id_r)
        self.M_test_dataset = Single_omics_dataset(self.MET_data, test_id_m)
        self.R_test_dataloader = DataLoader(self.R_test_dataset, batch_size=64, shuffle=False, num_workers=4)
        self.M_test_dataloader = DataLoader(self.M_test_dataset, batch_size=64, shuffle=False, num_workers=4)

        self.set_eval()
        my_logger.info('get predicting ...')
        """ record the predicted data """
        R2M_predict = []
        M2R_predict = []
        with torch.no_grad():
            with tqdm(total = len(self.R_test_dataloader), ncols=100) as pbar:
                pbar.set_description('RNA to MET predicting...')
                for idx, batch_samples in enumerate(self.R_test_dataloader):
                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    R2 = self.RNA_encoder(batch_samples)
                    R2R, R2M, mu_r, sigma_r = self.translator.test_model(R2, 'RNA')
                    R2A = self.MET_decoder(R2M)

                    R2M_predict.append(R2M.cpu())

                    time.sleep(0.01)
                    pbar.update(1)

        with torch.no_grad():
            with tqdm(total = len(self.M_test_dataloader), ncols=100) as pbar:
                pbar.set_description('MET to RNA predicting...')
                for idx, batch_samples in enumerate(self.M_test_dataloader):
                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    M2 = self.MET_encoder(batch_samples)
                    M2R, M2M, mu_m, sigma_m = self.translator.test_model(M2, 'MET')
                    M2R = self.RNA_decoder(M2R)

                    M2R_predict.append(M2R.cpu())
                    
                    time.sleep(0.01)
                    pbar.update(1)
        
        R2M_predict = tensor2adata(R2M_predict)
        M2R_predict = tensor2adata(M2R_predict)
        
        M2R_predict.obs = self.MET_data_obs.iloc[test_id_m, :]
        R2M_predict.obs = self.RNA_data_obs.iloc[test_id_r, :]

        """ draw umap if needed """
        if test_figure:
            my_logger.info('drawing tsne figures ...')
            fig_M2R = draw_tsne(M2R_predict, 'm2r', 'cell_type')
            fig_R2M = draw_tsne(R2M_predict, 'r2m', 'cell_type')
            
            fig_list = [fig_M2R, fig_R2M]
            with PdfPages(output_path + '/tSNE.pdf') as pdf:
                for i in range(len(fig_list)):
                    pdf.savefig(figure=fig_list[i], dpi=200, bbox_inches='tight')
                    plt.close()
        else:
            my_logger.info('calculate neighbors graph for following test ...')
            sc.pp.pca(M2R_predict)
            sc.pp.neighbors(M2R_predict)
            sc.pp.pca(R2M_predict)
            sc.pp.neighbors(R2M_predict)
                    
        """ test cluster index if needed """
        if test_cluster:
            index_R2M = calculate_cluster_index(R2M_predict)
            index_M2R = calculate_cluster_index(M2R_predict)
            
            index_matrix = pd.DataFrame([index_R2M, index_M2R])
            index_matrix.columns = ['ARI', 'AMI', 'NMI', 'HOM']
            index_matrix.index = ['R2M', 'M2R']
            index_matrix.to_csv(output_path + '/cluster_index.csv')
            

        """ save predicted model if needed """
        if output_data and not os.path.exists(output_path + '/predict'):
            my_logger.warning('trying to write predict to path: '+str(output_path)+'/predict')
            os.mkdir(output_path + '/predict')
            M2R_predict.write_h5ad(output_path + '/predict/M2R.h5ad')
            R2M_predict.write_h5ad(output_path + '/predict/R2M.h5ad')
            
        if return_predict:
            return M2R_predict, R2M_predict

    def predict_single_rna(
        self,
        test_id_r=None,
        model_path=None,
        load_model=True,
        output_path=None,
        batch_size=64,
        return_embeddings=False
    ):
        """
        Predict methylation data from RNA data using existing test indices.
        
        Parameters
        ----------
        test_id_r: list
            List of RNA test indices. If None, will use all RNA data.
            
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
        my_logger = create_logger(name='SingleModalPredictor', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
        my_logger.info('predicting methylation from RNA data...')
        
        # Load model if needed
        if load_model:
            if model_path is None:
                raise ValueError("model_path must be provided when load_model=True")
            my_logger.info('loading trained model from path: ' + str(model_path) + '/model')
            self.RNA_encoder.load_state_dict(torch.load(model_path + '/model/RNA_encoder.pt'))
            self.MET_decoder.load_state_dict(torch.load(model_path + '/model/MET_decoder.pt'))
            self.translator.load_state_dict(torch.load(model_path + '/model/translator.pt'))
        
        # Use test_id_r if provided, otherwise use all RNA data
        if test_id_r is None:
            test_id_r = list(range(len(self.RNA_data)))
        
        # Create dataset and dataloader using existing RNA data
        rna_dataset = Single_omics_dataset(self.RNA_data, test_id_r)
        rna_dataloader = DataLoader(rna_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        self.set_eval()
        
        # Prediction
        predicted_methylation = []
        latent_embeddings = []
        
        with torch.no_grad():
            with tqdm(total=len(rna_dataloader), ncols=100) as pbar:
                pbar.set_description('RNA to Methylation predicting...')
                for idx, batch_samples in enumerate(rna_dataloader):
                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)
                    
                    # Encode RNA data
                    rna_encoded = self.RNA_encoder(batch_samples)
                    
                    # Translate to methylation space
                    r2r, r2m, mu_r, sigma_r = self.translator.test_model(rna_encoded, 'RNA')
                    
                    # Decode to methylation data
                    predicted_met = self.MET_decoder(r2m)
                    
                    predicted_methylation.append(predicted_met.cpu())
                    
                    if return_embeddings:
                        latent_embeddings.append(r2m.cpu())
                    
                    pbar.update(1)
        
        # Convert to AnnData
        predicted_methylation = tensor2adata(predicted_methylation)
        predicted_methylation.obs = self.RNA_data_obs.iloc[test_id_r, :]
        
        # Save results if output_path is provided
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            predicted_methylation.write_h5ad(output_path + '/predicted_methylation.h5ad')
            my_logger.info('Predicted methylation data saved to: ' + output_path + '/predicted_methylation.h5ad')
        
        if return_embeddings:
            latent_embeddings = tensor2adata(latent_embeddings)
            latent_embeddings.obs = self.RNA_data_obs.iloc[test_id_r, :]
            return predicted_methylation, latent_embeddings
        
        return predicted_methylation

    def predict_single_methylation(
        self,
        test_id_m=None,
        model_path=None,
        load_model=True,
        output_path=None,
        batch_size=64,
        return_embeddings=False
    ):
        """
        Predict RNA data from methylation data using existing test indices.
        
        Parameters
        ----------
        test_id_m: list
            List of methylation test indices. If None, will use all methylation data.
            
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
        my_logger = create_logger(name='SingleModalPredictor', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
        my_logger.info('predicting RNA from methylation data...')
        
        if load_model:
            if model_path is None:
                raise ValueError("model_path must be provided when load_model=True")
            my_logger.info('loading trained model from path: ' + str(model_path) + '/model')
            self.MET_encoder.load_state_dict(torch.load(model_path + '/model/MET_encoder.pt'))
            self.RNA_decoder.load_state_dict(torch.load(model_path + '/model/RNA_decoder.pt'))
            self.translator.load_state_dict(torch.load(model_path + '/model/translator.pt'))
        
        if test_id_m is None:
            test_id_m = list(range(len(self.MET_data)))
        
        met_dataset = Single_omics_dataset(self.MET_data, test_id_m)
        met_dataloader = DataLoader(met_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        self.set_eval()
        
        predicted_rna = []
        latent_embeddings = []
        
        with torch.no_grad():
            with tqdm(total=len(met_dataloader), ncols=100) as pbar:
                pbar.set_description('Methylation to RNA predicting...')
                for idx, batch_samples in enumerate(met_dataloader):
                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)
                    
                    met_encoded = self.MET_encoder(batch_samples)
                    
                    m2m, m2r, mu_m, sigma_m = self.translator.test_model(met_encoded, 'MET')

                    predicted_rna_batch = self.RNA_decoder(m2r)
                    
                    predicted_rna.append(predicted_rna_batch.cpu())
                    
                    if return_embeddings:
                        latent_embeddings.append(m2r.cpu())
                    
                    pbar.update(1)
        
        predicted_rna = tensor2adata(predicted_rna)
        predicted_rna.obs = self.MET_data_obs.iloc[test_id_m, :]
        
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            predicted_rna.write_h5ad(output_path + '/predicted_rna.h5ad')
            my_logger.info('Predicted RNA data saved to: ' + output_path + '/predicted_rna.h5ad')
        
        if return_embeddings:
            latent_embeddings = tensor2adata(latent_embeddings)
            latent_embeddings.obs = self.MET_data_obs.iloc[test_id_m, :]
            return predicted_rna, latent_embeddings
        
        return predicted_rna