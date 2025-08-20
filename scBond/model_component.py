import torch
import torch.nn as nn
import torch.nn.functional as F


class NetBlock(nn.Module):
    def __init__(
        self,
        nlayer: int, 
        dim_list: list, 
        act_list: list, 
        dropout_rate: float, 
        noise_rate: float
        ):
        """
        multiple layers netblock with specific layer counts, dimension, activations and dropout.
        
        Parameters
        ----------
        nlayer
            layer counts.
            
        dim_list
            dimension list, length equal to nlayer + 1.
        
        act_list
            activation list, length equal to nlayer + 1.
        
        dropout_rate
            rate of dropout.
        
        noise_rate
            rate of set part of input data to 0.
            
        """
        super(NetBlock, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        
        for i in range(nlayer):
            
            self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            nn.init.xavier_uniform_(self.linear_list[i].weight)
            self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activation_list.append(act_list[i])
            if not i == nlayer -1: 
                self.dropout_list.append(nn.Dropout(dropout_rate))
        
    
    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            if not i == self.nlayer -1:
                """ don't use dropout for output to avoid loss calculate break down """
                x = self.dropout_list[i](x)

        return x


class Split_Chrom_Encoder_block(nn.Module):
    def __init__(
        self,
        nlayer: int, 
        dim_list: list, 
        act_list: list,
        chrom_list: list,
        dropout_rate: float, 
        noise_rate: float
        ):
        """
        MET encoder netblock with specific layer counts, dimension, activations and dropout.
        
        Parameters
        ----------
        nlayer
            layer counts.
            
        dim_list
            dimension list, length equal to nlayer + 1.
        
        act_list
            activation list, length equal to nlayer + 1.
            
        chrom_list
            list record the peaks count for each chrom, assert that sum of chrom list equal to dim_list[0].
            
        dropout_rate
            rate of dropout.
        
        noise_rate
            rate of set part of input data to 0.
            
        """
        super(Split_Chrom_Encoder_block, self).__init__()
        self.nlayer = nlayer
        self.chrom_list = chrom_list
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        
        for i in range(nlayer):
            if i == 0:
                """first layer seperately forward for each chrom"""
                self.linear_list.append(nn.ModuleList())
                self.bn_list.append(nn.ModuleList())
                self.activation_list.append(nn.ModuleList())
                self.dropout_list.append(nn.ModuleList())
                for j in range(len(chrom_list)):
                    self.linear_list[i].append(nn.Linear(chrom_list[j], dim_list[i + 1] // len(chrom_list)))
                    nn.init.xavier_uniform_(self.linear_list[i][j].weight)
                    self.bn_list[i].append(nn.BatchNorm1d(dim_list[i + 1] // len(chrom_list)))
                    self.activation_list[i].append(act_list[i])
                    self.dropout_list[i].append(nn.Dropout(dropout_rate))
            else:
                self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
                nn.init.xavier_uniform_(self.linear_list[i].weight)
                self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
                self.activation_list.append(act_list[i])
                if not i == nlayer -1: 
                    self.dropout_list.append(nn.Dropout(dropout_rate))
        
    
    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            if i == 0:
                x = torch.split(x, self.chrom_list, dim = 1)
                temp = []
                for j in range(len(self.chrom_list)):
                    temp.append(self.dropout_list[0][j](self.activation_list[0][j](self.bn_list[0][j](self.linear_list[0][j](x[j])))))
                x = torch.concat(temp, dim = 1)
            else:
                x = self.linear_list[i](x)
                x = self.bn_list[i](x)
                x = self.activation_list[i](x)
                if not i == self.nlayer -1:
                    """ don't use dropout for output to avoid loss calculate break down """
                    x = self.dropout_list[i](x)
        return x


class Split_Chrom_Decoder_block(nn.Module):
    def __init__(
        self,
        nlayer: int, 
        dim_list: list, 
        act_list: list,
        chrom_list: list,
        dropout_rate: float, 
        noise_rate: float
        ):
        """
        MET decoder netblock with specific layer counts, dimension, activations and dropout.
        
        Parameters
        ----------
        nlayer
            layer counts.
            
        dim_list
            dimension list, length equal to nlayer + 1.
        
        act_list
            activation list, length equal to nlayer + 1.
            
        chrom_list
            list record the peaks count for each chrom, assert that sum of chrom list equal to dim_list[end].
            
        dropout_rate
            rate of dropout.
        
        noise_rate
            rate of set part of input data to 0.
            
        """
        super(Split_Chrom_Decoder_block, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.chrom_list = chrom_list
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        
        for i in range(nlayer):
            if not i == nlayer -1:
                self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
                nn.init.xavier_uniform_(self.linear_list[i].weight)
                self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
                self.activation_list.append(act_list[i])
                self.dropout_list.append(nn.Dropout(dropout_rate))
            else:
                """last layer seperately forward for each chrom"""
                self.linear_list.append(nn.ModuleList())
                self.bn_list.append(nn.ModuleList())
                self.activation_list.append(nn.ModuleList())
                self.dropout_list.append(nn.ModuleList())
                for j in range(len(chrom_list)):
                    self.linear_list[i].append(nn.Linear(dim_list[i] // len(chrom_list), chrom_list[j]))
                    nn.init.xavier_uniform_(self.linear_list[i][j].weight)
                    self.bn_list[i].append(nn.BatchNorm1d(chrom_list[j]))
                    self.activation_list[i].append(act_list[i])
        
    
    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            if not i == self.nlayer -1:
                x = self.linear_list[i](x)
                x = self.bn_list[i](x)
                x = self.activation_list[i](x)
                x = self.dropout_list[i](x)
            else:
                x = torch.chunk(x, len(self.chrom_list), dim = 1)
                temp = []
                for j in range(len(self.chrom_list)):
                    temp.append(self.activation_list[i][j](self.bn_list[i][j](self.linear_list[i][j](x[j]))))
                x = torch.concat(temp, dim = 1)

        return x
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  
            nn.Sigmoid() 
        )

    def forward(self, x):
        b, c = x.size() 
        y = self.fc(x)  
        return x * y.view(b, c)  

    
class SelfAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=False, 
        attn_drop=0., 
        proj_drop=0.
    ):
        """
        Multi-head self-attention module that computes attention between input and context tensors.
    
        Parameters
        ----------
        dim: int
            Dimension of the input features and embedding.
        
        num_heads: int
            Number of parallel attention heads, default 8.
        
        qkv_bias: bool
            Whether to include bias in the query, key, and value linear projections, default False.
        
        attn_drop: float
            Dropout probability applied to attention weights, default 0.0.
        
        proj_drop: float
            Dropout probability applied to the output projection, default 0.0.
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(context).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Translator(nn.Module):
    def __init__(
        self, 
        translator_input_dim_r: int,
        translator_input_dim_m: int, 
        translator_embed_dim: int, 
        translator_embed_act_list: list, 
        num_experts: int = 4,        
        reduction_ratio: int = 64, 
        num_heads=8,
        attn_drop=0.1,
        proj_drop=0.1
    ):
        """
        Translator block with MoE mechanism used for translation between different omics.
        
        Parameters
        ----------
        translator_input_dim_r
            dimension of input from RNA encoder for translator.
            
        translator_input_dim_a
            dimension of input from MET encoder for translator.
            
        translator_embed_dim
            dimension of embedding space for translator.
            
        translator_embed_act_list
            activation list for translator, involving [mean_activation, log_var_activation, decoder_activation].
            
        num_experts
            number of expert networks. 
            
        reduction_ratio
            The reduction ratio for the SE layer.
            
        num_heads: int
            Number of parallel attention heads, default 8.
        
        attn_drop: float
            Dropout probability applied to attention weights, default 0.0.
        
        proj_drop: float
            Dropout probability applied to the output projection, default 0.0.
        """
        super(Translator, self).__init__()
        
        mean_activation, log_var_activation, decoder_activation = translator_embed_act_list
        
        self.expert_networks_r_mu = nn.ModuleList([
            nn.Sequential(
                nn.Linear(translator_input_dim_r, translator_embed_dim),
                nn.BatchNorm1d(translator_embed_dim),
                mean_activation
            ) for _ in range(num_experts)
        ])
        
        self.expert_networks_m_mu = nn.ModuleList([
            nn.Sequential(
                nn.Linear(translator_input_dim_m, translator_embed_dim),
                nn.BatchNorm1d(translator_embed_dim),
                mean_activation
            ) for _ in range(num_experts)
        ])
        
        self.expert_networks_r_d = nn.ModuleList([
            nn.Sequential(
                nn.Linear(translator_input_dim_r, translator_embed_dim),
                nn.BatchNorm1d(translator_embed_dim),
                log_var_activation
            ) for _ in range(num_experts)
        ])
        
        self.expert_networks_m_d = nn.ModuleList([
            nn.Sequential(
                nn.Linear(translator_input_dim_m, translator_embed_dim),
                nn.BatchNorm1d(translator_embed_dim),
                log_var_activation
            ) for _ in range(num_experts)
        ])
        
        self.gating_network_r_mu = nn.Sequential(
            nn.Linear(translator_input_dim_r, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )
        
        self.gating_network_m_mu = nn.Sequential(
            nn.Linear(translator_input_dim_m, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )
        
        self.gating_network_r_d = nn.Sequential(
            nn.Linear(translator_input_dim_r, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )
        
        self.gating_network_m_d = nn.Sequential(
            nn.Linear(translator_input_dim_m, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)
        )
        
        self.RNA_decoder_l = nn.Linear(translator_embed_dim, translator_input_dim_r)
        nn.init.xavier_uniform_(self.RNA_decoder_l.weight)
        self.RNA_decoder_bn = nn.BatchNorm1d(translator_input_dim_r)
        self.RNA_decoder_act = decoder_activation
        
        self.MET_decoder_l = nn.Linear(translator_embed_dim, translator_input_dim_m)
        nn.init.xavier_uniform_(self.MET_decoder_l.weight)
        self.MET_decoder_bn = nn.BatchNorm1d(translator_input_dim_m)
        self.MET_decoder_act = decoder_activation
        
        self.se_rna = SELayer(translator_embed_dim, reduction_ratio)
        self.se_met = SELayer(translator_embed_dim, reduction_ratio)
        
        self.self_attn_r2m = SelfAttention(
            dim=translator_embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        self.self_attn_m2r = SelfAttention(
            dim=translator_embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
    
    def reparameterize(self, mu, sigma):
        sigma = torch.exp(sigma / 2)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward_with_RNA(self, x, forward_type):

        expert_outputs_r_mu = [expert_network_mu(x) for expert_network_mu in self.expert_networks_r_mu]
        gating_weights_r_mu = self.gating_network_r_mu(x)
        gating_weights_r_mu = gating_weights_r_mu.unsqueeze(2)
        expert_outputs_r_mu = torch.stack(expert_outputs_r_mu, dim=1)
        weighted_output_r_mu = (expert_outputs_r_mu * gating_weights_r_mu).sum(dim=1)
        
        expert_outputs_r_d = [expert_network_d(x) for expert_network_d in self.expert_networks_r_d]
        gating_weights_r_d = self.gating_network_r_d(x)
        gating_weights_r_d = gating_weights_r_d.unsqueeze(2)
        expert_outputs_r_d = torch.stack(expert_outputs_r_d, dim=1)
        weighted_output_r_d = (expert_outputs_r_d * gating_weights_r_d).sum(dim=1)

        if forward_type == 'test':
            latent_layer = weighted_output_r_mu
        elif forward_type == 'train':
            latent_layer = self.reparameterize(weighted_output_r_mu, weighted_output_r_d)
        
        latent_layer = self.se_rna(latent_layer)
        
        latent_layer_R_out = self.RNA_decoder_act(self.RNA_decoder_bn(self.RNA_decoder_l(latent_layer)))
        latent_layer_M_out = self.MET_decoder_act(self.MET_decoder_bn(self.MET_decoder_l(latent_layer)))
        
        
        return latent_layer_R_out, latent_layer_M_out, weighted_output_r_mu, weighted_output_r_d


    def forward_with_MET(self, x, forward_type):
        
        expert_outputs_m_mu = [expert_network_mu(x) for expert_network_mu in self.expert_networks_m_mu]
        gating_weights_m_mu = self.gating_network_m_mu(x)
        gating_weights_m_mu = gating_weights_m_mu.unsqueeze(2)
        expert_outputs_m_mu = torch.stack(expert_outputs_m_mu, dim=1)
        weighted_output_m_mu = (expert_outputs_m_mu * gating_weights_m_mu).sum(dim=1)
        
        expert_outputs_m_d = [expert_network_d(x) for expert_network_d in self.expert_networks_m_d]
        gating_weights_m_d = self.gating_network_m_d(x)
        gating_weights_m_d = gating_weights_m_d.unsqueeze(2)
        expert_outputs_m_d = torch.stack(expert_outputs_m_d, dim=1)
        weighted_output_m_d = (expert_outputs_m_d * gating_weights_m_d).sum(dim=1)
        
        x = x.unsqueeze(1)
        weighted_output_m_mu = weighted_output_m_mu.unsqueeze(1)
        
        self_attn_output = self.self_attn_m2r(x, x)
        weighted_output_m_mu = weighted_output_m_mu + self_attn_output  # 残差连接
        
        weighted_output_m_mu = weighted_output_m_mu.squeeze(1)
        weighted_output_m_d = weighted_output_m_d.squeeze(1)
        
        if forward_type == 'test':
            latent_layer = weighted_output_m_mu
        elif forward_type == 'train':
            latent_layer = self.reparameterize(weighted_output_m_mu, weighted_output_m_d)
        
        latent_layer = self.se_met(latent_layer)
        
        latent_layer_R_out = self.RNA_decoder_act(self.RNA_decoder_bn(self.RNA_decoder_l(latent_layer)))
        latent_layer_M_out = self.MET_decoder_act(self.MET_decoder_bn(self.MET_decoder_l(latent_layer)))
        
        
        return latent_layer_R_out, latent_layer_M_out, weighted_output_m_mu, weighted_output_m_d

            
    def train_model(self, x, input_type):
        if input_type == 'RNA':
            return self.forward_with_RNA(x, 'train')
        elif input_type == 'MET':
            return self.forward_with_MET(x, 'train')
        
    def test_model(self, x, input_type):
        if input_type == 'RNA':
            return self.forward_with_RNA(x, 'test')
        elif input_type == 'MET':
            return self.forward_with_MET(x, 'test')


class Single_Translator(nn.Module):
    def __init__(
        self,
        translator_input_dim: int,  
        translator_embed_dim: int, 
        translator_embed_act_list: list, 
        num_experts_single: int = 6,      
        reduction_ratio: int = 64,   
        num_heads: int = 8,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1
        
    ):
        """
        Single translator block with MoE mechanism used only for pretraining.
        
        Parameters
        ----------
        translator_input_dim
            dimension of input from encoder for translator.
            
        translator_embed_dim
            dimension of embedding space for translator.
            
        translator_embed_act_list
            activation list for translator, involving [mean_activation, log_var_activation, decoder_activation].
            
        num_experts_single
            number of expert networks, default 6.
            
        reduction_ratio
            The reduction ratio for the SE layer, default 64.
            
        num_heads: int
            Number of parallel attention heads, default 8.
        
        attn_drop: float
            Dropout probability applied to attention weights, default 0.0.
        
        proj_drop: float
            Dropout probability applied to the output projection, default 0.0.
        """
        super(Single_Translator, self).__init__()
        
        mean_activation, log_var_activation, decoder_activation = translator_embed_act_list
        
        self.expert_networks_mu = nn.ModuleList([
            nn.Sequential(
                nn.Linear(translator_input_dim, translator_embed_dim),
                nn.BatchNorm1d(translator_embed_dim),
                mean_activation
            ) for _ in range(num_experts_single)
        ])
        
        self.expert_networks_d = nn.ModuleList([
            nn.Sequential(
                nn.Linear(translator_input_dim, translator_embed_dim),
                nn.BatchNorm1d(translator_embed_dim),
                log_var_activation
            ) for _ in range(num_experts_single)
        ])
        
        self.gating_network = nn.Sequential(
            nn.Linear(translator_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts_single),
            nn.Softmax(dim=1)
        )
        
        self.decoder_l = nn.Linear(translator_embed_dim, translator_input_dim)
        nn.init.xavier_uniform_(self.decoder_l.weight)
        self.decoder_bn = nn.BatchNorm1d(translator_input_dim)
        self.decoder_act = decoder_activation
        
        self.se = SELayer(translator_embed_dim, reduction_ratio)
        
        self.self_attn = SelfAttention(
            dim=translator_embed_dim,
            num_heads=8,
            attn_drop=0.1,
            proj_drop=0.1
        )
        
    def reparameterize(self, mu, sigma):
        sigma = torch.exp(sigma / 2)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward(self, x, forward_type):
        expert_outputs_mu = [expert_network(x) for expert_network in self.expert_networks_mu]
        expert_outputs_d = [expert_network(x) for expert_network in self.expert_networks_d]
    
        gating_weights = self.gating_network(x)
    
        gating_weights = gating_weights.unsqueeze(2)  
    
        expert_outputs_mu = torch.stack(expert_outputs_mu, dim=1)
    
        expert_outputs_d = torch.stack(expert_outputs_d, dim=1)
    
        weighted_output_mu = (expert_outputs_mu * gating_weights).sum(dim=1)
        weighted_output_d = (expert_outputs_d * gating_weights).sum(dim=1)
        
        weighted_output_mu = weighted_output_mu.unsqueeze(1)
        attn_output = self.self_attn(weighted_output_mu, weighted_output_mu)
        weighted_output_mu = weighted_output_mu + attn_output
        weighted_output_mu = weighted_output_mu.squeeze(1)
        
        if forward_type == 'test':
            latent_layer = weighted_output_mu
        elif forward_type == 'train':
            latent_layer = self.reparameterize(weighted_output_mu, weighted_output_d)
    
        latent_layer = self.se(latent_layer)
    
        latent_layer_out = self.decoder_act(self.decoder_bn(self.decoder_l(latent_layer)))
        return latent_layer_out, weighted_output_mu, weighted_output_d  

class Background_Translator(nn.Module):
    def __init__(
        self,
        translator_input_dim_r: int, 
        translator_input_dim_m: int, 
        translator_embed_dim: int, 
        translator_embed_act_list: list,  
        reduction_ratio: int = 64   
    ):
        """
        Background Translator block with SE layer.
        
        Parameters
        ----------
        translator_input_dim_r
            dimension of input from RNA encoder for translator.
            
        translator_input_dim_m
            dimension of input from MET encoder for translator.
            
        translator_embed_dim
            dimension of embedding space for translator.
            
        translator_embed_act_list
            activation list for translator, involving [mean_activation, log_var_activation, decoder_activation].
            
        reduction_ratio
            The reduction ratio for the SE layer, default 64.
        """
        super(Background_Translator, self).__init__()
        
        mean_activation, log_var_activation, decoder_activation = translator_embed_act_list
        
        self.RNA_encoder_l_mu = nn.Linear(translator_input_dim_r, translator_embed_dim)
        nn.init.xavier_uniform_(self.RNA_encoder_l_mu.weight)
        self.RNA_encoder_bn_mu = nn.BatchNorm1d(translator_embed_dim)
        self.RNA_encoder_act_mu = mean_activation
        
        self.MET_encoder_l_mu = nn.Linear(translator_input_dim_m, translator_embed_dim)
        nn.init.xavier_uniform_(self.MET_encoder_l_mu.weight)
        self.MET_encoder_bn_mu = nn.BatchNorm1d(translator_embed_dim)
        self.MET_encoder_act_mu = mean_activation
        
        self.RNA_encoder_l_d = nn.Linear(translator_input_dim_r, translator_embed_dim)
        nn.init.xavier_uniform_(self.RNA_encoder_l_d.weight)
        self.RNA_encoder_bn_d = nn.BatchNorm1d(translator_embed_dim)
        self.RNA_encoder_act_d = log_var_activation
        
        self.MET_encoder_l_d = nn.Linear(translator_input_dim_m, translator_embed_dim)
        nn.init.xavier_uniform_(self.MET_encoder_l_d.weight)
        self.MET_encoder_bn_d = nn.BatchNorm1d(translator_embed_dim)
        self.MET_encoder_act_d = log_var_activation
        
        self.RNA_decoder_l = nn.Linear(translator_embed_dim, translator_input_dim_r)
        nn.init.xavier_uniform_(self.RNA_decoder_l.weight)
        self.RNA_decoder_bn = nn.BatchNorm1d(translator_input_dim_r)
        self.RNA_decoder_act = decoder_activation
        
        self.MET_decoder_l = nn.Linear(translator_embed_dim, translator_input_dim_m)
        nn.init.xavier_uniform_(self.MET_decoder_l.weight)
        self.MET_decoder_bn = nn.BatchNorm1d(translator_input_dim_m)
        self.MET_decoder_act = decoder_activation
        
        self.background_pro_alpha = nn.Parameter(torch.randn(1, translator_input_dim_m))
        self.background_pro_log_beta = nn.Parameter(torch.clamp(torch.randn(1, translator_input_dim_m), -10, 1))
        
        self.scale_parameters_l = nn.Linear(translator_embed_dim, translator_embed_dim)
        self.scale_parameters_bn = nn.BatchNorm1d(translator_embed_dim)
        self.scale_parameters_act = mean_activation
        self.pi_l = nn.Linear(translator_embed_dim, 1)
        self.pi_bn = nn.BatchNorm1d(1)
        self.pi_act = nn.Sigmoid()
        
        self.se_rna = SELayer(translator_embed_dim, reduction_ratio)
        self.se_met = SELayer(translator_embed_dim, reduction_ratio)
    
    def reparameterize(self, mu, sigma):
        sigma = torch.exp(sigma / 2)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward_with_RNA(self, x, forward_type):
        latent_layer_mu = self.RNA_encoder_act_mu(self.RNA_encoder_bn_mu(self.RNA_encoder_l_mu(x)))
        latent_layer_d = self.RNA_encoder_act_d(self.RNA_encoder_bn_d(self.RNA_encoder_l_d(x)))
        
        if forward_type == 'test':
            latent_layer = latent_layer_mu
        elif forward_type == 'train':
            latent_layer = self.reparameterize(latent_layer_mu, latent_layer_d)
        
        latent_layer = self.se_rna(latent_layer)
        
        latent_layer_R_out = self.RNA_decoder_act(self.RNA_decoder_bn(self.RNA_decoder_l(latent_layer)))
        latent_layer_M_out = self.MET_decoder_act(self.MET_decoder_bn(self.MET_decoder_l(latent_layer)))

        return latent_layer_R_out, latent_layer_M_out, latent_layer_mu, latent_layer_d
    
    def forward_with_MET(self, x, forward_type):
        latent_layer = self.forward_adt(x, forward_type)
        
        latent_layer = self.se_met(latent_layer)
        
        latent_layer_R_out = self.RNA_decoder_act(self.RNA_decoder_bn(self.RNA_decoder_l(latent_layer)))
        latent_layer_M_out = self.MET_decoder_act(self.MET_decoder_bn(self.MET_decoder_l(latent_layer)))

        return latent_layer_R_out, latent_layer_M_out, latent_layer, latent_layer
    
    def train_model(self, x, input_type):
        if input_type == 'RNA':
            return self.forward_with_RNA(x, 'train')
        elif input_type == 'MET':
            return self.forward_with_MET(x, 'train')
        
    def test_model(self, x, input_type):
        if input_type == 'RNA':
            return self.forward_with_RNA(x, 'test')
        elif input_type == 'MET':
            return self.forward_with_MET(x, 'test')


class Background_Single_Translator(nn.Module):
    def __init__(
        self,
        translator_input_dim: int,  
        translator_embed_dim: int, 
        translator_embed_act_list: list, 
        reduction_ratio: int = 64    
    ):
        """
        Background Single Translator block with SE layer.
        
        Parameters
        ----------
        translator_input_dim
            dimension of input from encoder for translator.
            
        translator_embed_dim
            dimension of embedding space for translator.
            
        translator_embed_act_list
            activation list for translator, involving [mean_activation, log_var_activation, decoder_activation].
            
        reduction_ratio
            The reduction ratio for the SE layer, default 64.
        """
        super(Background_Single_Translator, self).__init__()
        
        mean_activation, log_var_activation, decoder_activation = translator_embed_act_list
        
        self.encoder_l_mu = nn.Linear(translator_input_dim, translator_embed_dim)
        nn.init.xavier_uniform_(self.encoder_l_mu.weight)
        self.encoder_bn_mu = nn.BatchNorm1d(translator_embed_dim)
        self.encoder_act_mu = mean_activation
        
        self.encoder_l_d = nn.Linear(translator_input_dim, translator_embed_dim)
        nn.init.xavier_uniform_(self.encoder_l_d.weight)
        self.encoder_bn_d = nn.BatchNorm1d(translator_embed_dim)
        self.encoder_act_d = log_var_activation

        self.decoder_l = nn.Linear(translator_embed_dim, translator_input_dim)
        nn.init.xavier_uniform_(self.decoder_l.weight)
        self.decoder_bn = nn.BatchNorm1d(translator_input_dim)
        self.decoder_act = decoder_activation
        
        self.background_pro_alpha = nn.Parameter(torch.randn(1, translator_input_dim))
        self.background_pro_log_beta = nn.Parameter(torch.clamp(torch.randn(1, translator_input_dim), -10, 1))
        
        self.scale_parameters_l = nn.Linear(translator_embed_dim, translator_embed_dim)
        self.scale_parameters_bn = nn.BatchNorm1d(translator_embed_dim)
        self.scale_parameters_act = mean_activation
        self.pi_l = nn.Linear(translator_embed_dim, 1)
        self.pi_bn = nn.BatchNorm1d(1)
        self.pi_act = nn.Sigmoid()
        
        self.se = SELayer(translator_embed_dim, reduction_ratio)
    
    def reparameterize(self, mu, sigma):
        sigma = torch.exp(sigma / 2)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward(self, x, forward_type):
        latent_layer = self.forward_adt(x, forward_type)
        
        latent_layer = self.se(latent_layer)
        
        latent_layer_out = self.decoder_act(self.decoder_bn(self.decoder_l(latent_layer)))
        return latent_layer_out, latent_layer, latent_layer