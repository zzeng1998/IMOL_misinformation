import copy
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import AutoConfig, BertModel
from transformers.models.bert.modeling_bert import BertLayer
# from zmq import device
from .coattention import *
from .layers import *
from utils.metrics import *
# from .RACL import retrieve_topk_batch_with_tensor
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import faiss
from easydict import EasyDict

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Create random tensors for image, text, video, and audio features, using the selected device (either CPU or GPU)
# device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
class ResidualAE(nn.Module):
    ''' Residual autoencoder using fc layers
        layers should be something like [128, 64, 32]
        eg:[128,64,32]-> add: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (64, 128), (128, input_dim)]
                          concat: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (128, 128), (256, input_dim)]
    '''

    def __init__(self, layers, n_blocks, input_dim, dropout=0.5, use_bn=False):
        super(ResidualAE, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.transition = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        for i in range(n_blocks):
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))

    def get_encoder(self, layers):
        all_layers = []
        input_dim = self.input_dim
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        # delete the activation layer of the last layer
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]
        return nn.Sequential(*all_layers)

    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(0, len(decoder_layer) - 2):
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i + 1]))
            all_layers.append(nn.ReLU())  # LeakyReLU
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))

        all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))
        return nn.Sequential(*all_layers)

    def forward(self, x):
        x_in = x
        x_out = x.clone().fill_(0)
        latents = []
        for i in range(self.n_blocks):
            encoder = getattr(self, 'encoder_' + str(i))
            decoder = getattr(self, 'decoder_' + str(i))
            x_in = x_in + x_out
            latent = encoder(x_in)
            x_out = decoder(latent)
            latents.append(latent)
        latents = torch.cat(latents, dim=-1)
        return self.transition(x_in + x_out), latents


def dense_retrieve_hard_negatives_pseudo_positive(
        query_feats, query_labels, train_feats, train_labels,
        largest_retrieval=5, no_hard_negatives=10, no_pseudo_gold_positives=2, metric="l2"
):
    """
    执行基于特征的密集检索并生成硬负样本和伪正样本。

    Args:
        query_feats (Tensor): 查询特征，大小 (batch_size, dim)
        query_labels (Tensor): 查询标签，大小 (batch_size,)
        train_feats (Tensor): 数据库特征，大小 (batch_size, dim)
        train_labels (Tensor): 数据库标签，大小 (batch_size,)
        largest_retrieval (int): 每个查询的最大检索数
        no_hard_negatives (int): 每个查询的硬负样本数
        no_pseudo_gold_positives (int): 每个查询的伪正样本数
        metric (str): 相似度度量方式，支持 "l2" 和 "ip"

    Returns:
        hard_negative_features (Tensor): 硬负样本特征，大小 (batch_size, no_hard_negatives, dim)
        pseudo_positive_features (Tensor): 伪正样本特征，大小 (batch_size, no_pseudo_gold_positives, dim)
    """
    dim = train_feats.shape[1]

    # 构建FAISS索引
    if metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)

    # 对训练数据进行归一化（如果使用L2度量）
    faiss.normalize_L2(train_feats.detach().cpu().numpy())
    faiss.normalize_L2(query_feats.detach().cpu().numpy())

    index.add(train_feats.detach().cpu().numpy())

    # 对查询进行检索，返回相似度和索引
    D, I = index.search(query_feats.detach().cpu().numpy(), largest_retrieval)

    # 初始化硬负样本和伪正样本
    hard_negative_features = torch.zeros(
        query_feats.shape[0], no_hard_negatives, dim, device=query_feats.device
    )
    pseudo_positive_features = torch.zeros(
        query_feats.shape[0], no_pseudo_gold_positives, dim, device=query_feats.device
    )

    # 遍历每个查询
    for i, row in enumerate(D):
        hard_negatives = 0
        pseudo_positives = 0

        for j, value in enumerate(row):
            # 检查是否是硬负样本（标签不同）
            if train_labels[I[i, j]] != query_labels[i] and hard_negatives < no_hard_negatives:
                hard_negative_features[i, hard_negatives] = train_feats[I[i, j]]
                hard_negatives += 1
                print("hard_negatives")
                print(j)
            # 检查是否是伪正样本（标签相同）
            elif train_labels[I[i, j]] == query_labels[i] and pseudo_positives < no_pseudo_gold_positives:
                pseudo_positive_features[i, pseudo_positives] = train_feats[I[i, j]]
                pseudo_positives += 1
                print("pseudo_positives")
                print(j)
            # 如果硬负样本和伪正样本都收集完，则跳出循环
            if hard_negatives == no_hard_negatives and pseudo_positives == no_pseudo_gold_positives:
                break

    return hard_negative_features, pseudo_positive_features



def nt_xent_loss(query_feats, pseudo_positive_feats, hard_negative_feats, temperature=0.05):

    batch_size = query_feats.size(0)

    # 选择第一个伪正样本作为正样本
    positive_feats = pseudo_positive_feats[:, 0, :]  # (batch_size, dim)

    # 计算查询与伪正样本的余弦相似度 (正样本相似度)
    positive_similarity = F.cosine_similarity(query_feats, positive_feats.mean(dim=1).unsqueeze(1), dim=-1)  # (batch_size,)

    # 计算查询与硬负样本的余弦相似度 (负样本相似度)
    negative_similarity = F.cosine_similarity(
        query_feats.unsqueeze(1), hard_negative_feats, dim=-1)  # (batch_size, no_hard_negatives)

    # 拼接正样本和负样本相似度
    logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity],
                       dim=1)  # (batch_size, 1 + no_hard_negatives)

    # 归一化 (softmax) 前除以温度
    logits /= temperature

    # 目标是让正样本（第0个位置）具有最高的概率
    labels = torch.zeros(batch_size, dtype=torch.long, device=query_feats.device)  # 正样本标签位置为0

    # 使用交叉熵计算损失
    loss = F.cross_entropy(logits, labels)
    return loss



def random_mask(view_num, input_len, missing_rate):
    """
    Randomly generates incomplete data (masking certain views), simulating partial views from complete data.

    Parameters:
    view_num (int): The total number of possible views (columns in the output matrix).
    input_len (int): The number of samples (rows in the output matrix).
    missing_rate (float): The fraction of missing views per sample (between 0 and 1).

    Returns:
    np.ndarray: A binary matrix of shape (input_len, view_num) where 1 indicates preserved views and
                0 indicates missing views.
    """
    # Ensure missing_rate is provided
    assert missing_rate is not None

    # Calculate the proportion of preserved views
    one_rate = 1 - missing_rate

    # Case 1: If one_rate is very small, only one view will be preserved per sample
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        # Randomly select a single view to preserve for each sample
        view_preserve = enc.fit_transform(np.random.randint(0, view_num, size=(input_len, 1))).toarray()
        return view_preserve  # Binary matrix where one view is preserved

    # Case 2: If all views are preserved (one_rate == 1)
    if one_rate == 1:
        matrix = np.ones((input_len, view_num), dtype=int)  # All ones (all views preserved)
        return matrix

    # Case 3: For one_rate between [1 / view_num, 1], we aim to preserve multiple views with certain randomness
    # Ensure at least 32 samples if input_len is too small
    alldata_len = max(input_len, 32)

    error = 1
    while error >= 0.005:  # Repeat until error is within acceptable tolerance
        # Generate initial view_preserve matrix with random views
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(np.random.randint(0, view_num, size=(alldata_len, 1))).toarray()

        # Compute the number of views to preserve
        one_num = view_num * alldata_len * one_rate - alldata_len  # Remaining number of views to preserve
        ratio = one_num / (view_num * alldata_len)  # Current ratio of preserved views

        # Generate a matrix with random views preserved based on the calculated ratio
        matrix_iter = (np.random.randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)

        # Count the overlaps where multiple views are selected for a sample
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)

        # Generate another matrix iteration based on the updated ratio
        matrix_iter = (np.random.randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)

        # Combine the generated matrix with the preserved views and apply a final masking
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)

        # Calculate the final ratio and error
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)

    # Slice the matrix to match the original input_len and return it
    matrix = matrix[:input_len, :]
    return matrix




class MemoryNetwork(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num=9, memory_num=10):
        super(MemoryNetwork, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32
        self.topic_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)

        # 将 domain_memory 替换为 nn.ModuleList 包含 Embedding
        self.domain_memory = nn.ModuleList([
            nn.Embedding(memory_num, emb_dim) for _ in range(domain_num)
        ])

    def forward(self, feature, category):
        feature = F.normalize(feature, p=2, dim=-1)  # Normalize feature vectors
        domain_label = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_memory = torch.stack([
            self.domain_memory[cat.item()](torch.arange(self.memory_num, device=feature.device)) for cat in category
        ])
        sep_domain_embedding = []
        for i in range(self.domain_num):
            topic_att = torch.nn.functional.softmax(torch.mm(self.topic_fc(feature), domain_memory[i].T) * self.tau,
                                                    dim=1)
            tmp_domain_embedding = torch.mm(topic_att, domain_memory[i])
            sep_domain_embedding.append(tmp_domain_embedding.unsqueeze(1))
        sep_domain_embedding = torch.cat(sep_domain_embedding, 1)
        domain_att = torch.bmm(sep_domain_embedding, self.domain_fc(feature).unsqueeze(2)).squeeze()

        domain_att = torch.nn.functional.softmax(domain_att * self.tau, dim=1).unsqueeze(1)

        return domain_att

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, activation=F.relu, dropout=0.0):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # Define the hidden layers
        layers = []
        if num_layers > 1:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Forward pass through all layers
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        # Last layer without activation
        x = self.layers[-1](x)
        return x



class IMOLModel(torch.nn.Module):
    def __init__(self, bert_model, fea_dim, dropout):
        super(IMOLModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model).requires_grad_(False)

        self.text_dim = 768
        self.comment_dim = 768
        self.img_dim = 4096
        self.audio_dim = 12288
        self.video_dim = 4096
        self.fea_dim = fea_dim
        self.num_heads = 4
        self.dropout = dropout

        self.num_frames = 83
        self.num_audioframes = 50
        self.num_comments = 23
        self.dim = fea_dim
        self.domain_num = 9
        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=128)
        self.vggish_layer = torch.hub.load('/data/kxz/FakeSV-main/code/torchvggish/', 'vggish', source='local')
        net_structure = list(self.vggish_layer.children())
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])

        self.co_attention_ta = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=fea_dim,
                                            visual_len=self.num_audioframes, sen_len=512, fea_v=self.dim,
                                            fea_s=self.dim, pos=False)
        self.co_attention_tv = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=fea_dim,
                                            visual_len=self.num_frames, sen_len=512, fea_v=self.dim, fea_s=self.dim,
                                            pos=False)
        self.trm = nn.TransformerEncoderLayer(d_model=self.dim, nhead=2, batch_first=True)

        self.memory_network = MemoryNetwork(input_dim=fea_dim, emb_dim=fea_dim, domain_num=9,
                                            memory_num=10)
        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim), torch.nn.ReLU(),
                                         nn.Dropout(p=self.dropout))
        self.linear_comment = nn.Sequential(torch.nn.Linear(self.comment_dim, fea_dim), torch.nn.ReLU(),
                                            nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim), torch.nn.ReLU(),
                                        nn.Dropout(p=self.dropout))
        self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, fea_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))
        self.linear_intro = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(fea_dim, fea_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))

        self.linear_memory = nn.Sequential(torch.nn.Linear(fea_dim*2, fea_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))

        # Input Projections
        self.text_proj = nn.Sequential(nn.Linear(self.fea_dim, fea_dim), nn.ReLU(), nn.Dropout(dropout))
        self.comment_proj = nn.Sequential(nn.Linear(self.fea_dim, fea_dim), nn.ReLU(), nn.Dropout(dropout))
        self.img_proj = nn.Sequential(nn.Linear(self.fea_dim, fea_dim), nn.ReLU(), nn.Dropout(dropout))
        self.audio_proj = nn.Sequential(nn.Linear(self.fea_dim, fea_dim), nn.ReLU(), nn.Dropout(dropout))
        self.video_proj = nn.Sequential(nn.Linear(self.fea_dim, fea_dim), nn.ReLU(), nn.Dropout(dropout))
        self.text_DE_proj = nn.Sequential(nn.Linear(self.fea_dim, fea_dim), nn.ReLU(), nn.Dropout(dropout))
        self.video_DE_proj = nn.Sequential(nn.Linear(self.fea_dim, fea_dim), nn.ReLU(), nn.Dropout(dropout))
        self.memory_proj = nn.Sequential(nn.Linear(self.fea_dim, fea_dim), nn.ReLU(), nn.Dropout(dropout))
        self.commemory_proj = nn.Sequential(nn.Linear(self.fea_dim, fea_dim), nn.ReLU(), nn.Dropout(dropout))
        # Router Modules (Dynamic Routing for Each Modality)
        self.text_router = nn.Sequential(nn.Linear(fea_dim, fea_dim), nn.ReLU(), nn.Linear(fea_dim, 6),nn.Softmax(dim=-1))
        self.img_router = nn.Sequential(nn.Linear(fea_dim, fea_dim), nn.ReLU(), nn.Linear(fea_dim, 6),nn.Softmax(dim=-1))
        self.audio_router = nn.Sequential(nn.Linear(fea_dim, fea_dim), nn.ReLU(), nn.Linear(fea_dim, 6),nn.Softmax(dim=-1))
        self.video_router = nn.Sequential(nn.Linear(fea_dim, fea_dim), nn.ReLU(), nn.Linear(fea_dim, 6),nn.Softmax(dim=-1))
        self.text_DE_router = nn.Sequential(nn.Linear(fea_dim, fea_dim), nn.ReLU(), nn.Linear(fea_dim, 6),nn.Softmax(dim=-1))
        self.video_DE_router = nn.Sequential(nn.Linear(fea_dim, fea_dim), nn.ReLU(), nn.Linear(fea_dim, 6),nn.Softmax(dim=-1))
        self.memory_router = nn.Sequential(nn.Linear(fea_dim, fea_dim), nn.ReLU(), nn.Linear(fea_dim, 6),nn.Softmax(dim=-1))
        self.commemory_router = nn.Sequential(nn.Linear(fea_dim, fea_dim), nn.ReLU(), nn.Linear(fea_dim, 6),nn.Softmax(dim=-1))
        # Transformer Layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=fea_dim, nhead=4, dropout=dropout, batch_first=True),
            num_layers=2,
        )
        # Contrastive learning
        self.criterion_mse = torch.nn.MSELoss()
        AE_layers = '768,128,32'
        AE_layers = list(map(lambda x: int(x), AE_layers.split(',')))
        self.netAE = ResidualAE(AE_layers, 3, 768, dropout=0.5, use_bn=False)
        self.netAE_cycle = ResidualAE(AE_layers, 3, 768, dropout=0.5, use_bn=False)
        # Feature Fusion
        self.fusion = nn.Sequential(nn.Linear(fea_dim * 3, fea_dim), nn.ReLU(), nn.Dropout(dropout))

        # Output Layers
        self.classifier = nn.Linear(fea_dim, 2)
        self.mlp = MLP(input_dim = fea_dim, hidden_dim=64, output_dim=2, num_layers=3, dropout=0.2)
        self.masked_matrix = random_mask(view_num=4, input_len=128, missing_rate=0.7)
        self.encoder1 = nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), nn.Dropout(p=self.dropout))
        self.encoder2 = nn.Sequential(torch.nn.Linear(128, 128), torch.nn.ReLU(), nn.Dropout(p=self.dropout))


    def forward(self,  **kwargs):



        ### Title ###
        title_inputid = kwargs['title_inputid']#(batch,512)
        title_mask=kwargs['title_mask']#(batch,512)

        fea_text=self.bert(title_inputid,attention_mask=title_mask)['last_hidden_state']#(batch,sequence,768)
        fea_text=self.linear_text(fea_text)

        ### Audio Frames ###
        audioframes=kwargs['audioframes']#(batch,36,12288)
        audioframes_masks = kwargs['audioframes_masks']
        fea_audio = self.vggish_modified(audioframes) #(batch, frames, 128)
        fea_audio = self.linear_audio(fea_audio)
        fea_audio, fea_text = self.co_attention_ta(v=fea_audio, s=fea_text, v_len=fea_audio.shape[1], s_len=fea_text.shape[1])
        fea_audio = torch.mean(fea_audio, -2)

        ### Image Frames ###
        frames=kwargs['frames']#(batch,30,4096)
        frames_masks = kwargs['frames_masks']
        fea_img = self.linear_img(frames)
        fea_img, fea_text = self.co_attention_tv(v=fea_img, s=fea_text, v_len=fea_img.shape[1], s_len=fea_text.shape[1])

        ### Title ###


        fea_img = torch.mean(fea_img, -2)

        fea_text = torch.mean(fea_text, -2)

        ### C3D ###
        c3d = kwargs['c3d'] # (batch, 36, 4096)
        c3d_masks = kwargs['c3d_masks']
        fea_video = self.linear_video(c3d) #(batch, frames, 128)
        fea_video = torch.mean(fea_video, -2)
        print("self.masked_matrix")
        print(self.masked_matrix)

        batch_size = fea_text.size(0)
        text_mask = np.reshape(self.masked_matrix[:batch_size, 0], ( batch_size, 1))
        text_mask = torch.LongTensor(text_mask).cuda()
        fea_text = fea_text * text_mask
        image_mask = np.reshape(self.masked_matrix[:batch_size, 1], (batch_size, 1))
        image_mask = torch.LongTensor(image_mask).cuda()
        fea_img = fea_img * image_mask


        video_mask = np.reshape(self.masked_matrix[:batch_size, 2], (batch_size, 1))
        video_mask = torch.LongTensor(video_mask).cuda()
        fea_video = fea_video * video_mask

        audio_mask = np.reshape(self.masked_matrix[:batch_size, 3], (batch_size, 1))
        audio_mask = torch.LongTensor(audio_mask).cuda()
        fea_audio = fea_audio * audio_mask

        # comment_mask = np.reshape(self.masked_matrix[:batch_size, 4], (batch_size, 1))
        # comment_mask = torch.LongTensor(comment_mask).cuda()

        label = kwargs['label']

        batch_size = fea_text.size(0)
        category = kwargs['type']
        fea_text = fea_text.unsqueeze(1)

        fea_img = fea_img.unsqueeze(1)
        fea_audio = fea_audio.unsqueeze(1)
        fea_video = fea_video.unsqueeze(1)



        fea = torch.cat((fea_text, fea_audio, fea_video, fea_img),1)  # (bs, 6, 128)

        # Apply RACL
        database = (np.arange(batch_size), fea.mean(dim=1), label)
        hard_negative_feats, pseudo_positive_feats = dense_retrieve_hard_negatives_pseudo_positive(
            query_feats=database[1], query_labels=database[2],
            train_feats=database[1], train_labels=database[2]
        )
        # 计算对比损失
        loss = nt_xent_loss(database[1], pseudo_positive_feats, hard_negative_feats, temperature=0.06)#temperature=0.05,0.02,0.01,0.03 -2,3,4,1,0
        # loss = info_nce_loss(database[1], pseudo_positive_feats, hard_negative_feats, temperature=0.05)
        # loss = contrastive_loss(database[1], database[2],database[1], database[2],
        #                         hard_negative_feats, pseudo_positive_feats, margin=1.0)
        # Apply Memory Network
        memory_att = self.memory_network(fea.mean(dim=1), category)
        domain_emb_all = self.domain_embedder(torch.LongTensor(range(self.domain_num)).cuda())

        general_domain_embedding = torch.mm(memory_att.squeeze(1), domain_emb_all)

        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embedding = self.domain_embedder(idxs).squeeze(1)
        # gate_input = torch.cat([domain_embedding, general_domain_embedding], dim=-1)
        # domain_embedding = self.linear_memory(torch.cat((general_domain_embedding,domain_embedding),1))
        # Input Projections
        text_fea = self.text_proj(fea_text)
        img_fea = self.img_proj(fea_img)
        audio_fea = self.audio_proj(fea_audio)
        video_fea = self.video_proj(fea_video)


        memory_fea = self.memory_proj(domain_embedding.unsqueeze(1))

        general_memory_fea = self.commemory_proj(general_domain_embedding.unsqueeze(1))
        # Dynamic Routing for Each Modality
        text_weights = self.text_router(text_fea).unsqueeze(-1)  # (batch, seq, 3, 1)
        img_weights = self.img_router(img_fea).unsqueeze(-1)
        audio_weights = self.audio_router(audio_fea).unsqueeze(-1)
        video_weights = self.video_router(video_fea).unsqueeze(-1)
        memory_weights = self.memory_router(memory_fea).unsqueeze(-1)
        general_memory_weights = self.commemory_router(general_memory_fea).unsqueeze(-1)
        # Weighted Feature Aggregation
        text_fea = torch.sum(text_weights * text_fea.unsqueeze(-2), dim=-2)
        img_fea = torch.sum(img_weights * img_fea.unsqueeze(-2), dim=-2)
        audio_fea = torch.sum(audio_weights * audio_fea.unsqueeze(-2), dim=-2)
        video_fea = torch.sum(video_weights * video_fea.unsqueeze(-2), dim=-2)

        memory_fea = torch.sum(memory_weights * memory_fea.unsqueeze(-2), dim=-2)
        general_memory_fea = torch.sum(general_memory_weights * memory_fea.unsqueeze(-2), dim=-2)
        memory_fea = fea.mean(dim=1).unsqueeze(1) * memory_fea
        general_memory_fea = fea.mean(dim=1).unsqueeze(1) * general_memory_fea
        # Combine All Modalities
        fea = torch.cat((text_fea, audio_fea, video_fea, img_fea), 1)  # (bs, 6, 128)
        combined_fea = torch.concat([fea.mean(dim=1).unsqueeze(1),memory_fea, general_memory_fea], dim=1)
        # feat_fusion_miss = combined_fea.mean(dim=1)
        feat_fusion_miss = torch.cat((text_fea.squeeze(1), audio_fea.squeeze(1), video_fea.squeeze(1), img_fea.squeeze(1), memory_fea.squeeze(1), general_memory_fea.squeeze(1)), 1)
        recon_cycle, latent = self.netAE_cycle(feat_fusion_miss)
        loss_cycle = self.criterion_mse(feat_fusion_miss.detach(),recon_cycle)
        # Transformer Extraction
        transformer_output = self.transformer(combined_fea)

        # Feature Fusion
        fused_fea = self.fusion(transformer_output.view(batch_size, -1))

        # Final Output
        output = self.classifier(fused_fea)

        return output, fused_fea,loss, loss_cycle
  