import torch
from torch import nn
from torch_geometric.nn import GCNConv, TransformerConv
import numpy as np

#torch.backends.cudnn.enabled = False

class intra_domain_encoder(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.GCN = GCNConv(dim, dim)
        self.GTrans = TransformerConv(dim, dim, heads=n_heads,concat=False, beta=True)
        self.CNN = nn.Conv1d(in_channels=1,
                               out_channels=dim,
                               kernel_size=(dim, 1),
                               stride=1,
                               bias=True)
        
    def forward(self, mole, f_mole, data):
        n_mole = f_mole.shape[0]
        f_gcn = torch.relu(self.GCN(f_mole.cuda(), data[mole]['edges'].cuda(), data[mole]['matrix'][data[mole]['edges'][0], data[mole]['edges'][1]].cuda()))
        f_tf= torch.relu(self.GTrans(f_gcn,data[mole]['edges'].cuda()))
        f_tf = f_tf.t().view(1, 1, -1, n_mole)
        f_intra = self.CNN(f_tf)
        f_intra = f_intra.view(1, n_mole, -1)
        return f_intra


class inter_domain_encoder(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.GCN = GCNConv(dim, dim)
        self.GTrans = TransformerConv(dim, dim, heads=n_heads,concat=False, beta=True)
        self.CNN = nn.Conv1d(in_channels=1,
                               out_channels=dim,
                               kernel_size=(dim, 1),
                               stride=1,
                               bias=True)
        
    def forward(self, f_all, data):
        n_mole = f_all.shape[0]   
        inter_gcn = torch.relu(self.GCN(f_all.cuda(), data['Inter'].type(torch.long).cuda(),None))
        inter_tf =torch.relu(self.GTrans(inter_gcn, data['Inter'].type(torch.long).cuda()))
        inter_tf = inter_tf.t().view(1, 1, -1, n_mole)
        f_inter = self.CNN(inter_tf)
        f_inter = f_inter.view(1, n_mole, -1)
        return f_inter
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    def forward(self, Q, K, V):
        #Q: [batch_size, n_heads, len_q, d_k]
        #K: [batch_size, n_heads, len_k, d_k]
        #V: [batch_size, n_heads, len_v(=len_k), d_v]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) #scores:[batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) #[batch_size, n_heads, len_q, d_v]
        return context
    

class multihead_cross_attention(nn.Module):
    def __init__(self, d_model, n_heads, d_k):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.W_Q = nn.Conv1d(d_model, d_k*n_heads, kernel_size=1, stride=1, padding=0)
        self.W_K = nn.Conv1d(d_model, d_k*n_heads, kernel_size=1, stride=1, padding=0)
        self.W_V = nn.Conv1d(d_model, d_k*n_heads, kernel_size=1, stride=1, padding=0)
        self.ScaledDotProductAttention = ScaledDotProductAttention(d_k)
        self.FFN = PoswiseFeedForwardNet(d_model, d_model*2)
        self.fc = nn.Linear(n_heads*d_k, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        batch_size = input_Q.size(0)
        '''residual, batch_size = input_Q, input_Q.size(0)'''
        #(B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q.transpose(1, 2)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #Q:[bs, heads, len_q, d_k]
        K = self.W_K(input_K.transpose(1, 2)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #K:[bs, heads, len_k, d_k]
        V = self.W_V(input_V.transpose(1, 2)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #V:[bs, heads, len_v(=len_k), d_v]
        #print(Q.shape, K.shape, V.shape)
        #context:[batch_size, n_heads, len_q, d_v]
        #attn:[batch_size, n_heads, len_q, len_k]
        context = self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads*self.d_k) #context:[bs, len_q, heads*d_v]
        output = self.fc(context) #[batch_size, len_q, d_model]

        return output


class cross_domain_decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k):
        super().__init__()
        self.decoder = multihead_cross_attention(d_model, n_heads, d_k)

    def forward(self, f_x_intra, f_x_inter):
        inter2intra = self.decoder(f_x_intra,f_x_inter,f_x_inter)
        intra2inter = self.decoder(f_x_inter,f_x_intra,f_x_intra)
        f_decoded = torch.cat([inter2intra, intra2inter], dim=-1)
        return f_decoded


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_in, d_ff):
        super().__init__()
        self.d_in = d_in
        self.d_ff = d_ff

    def forward(self, inputs):
        #inputs:[batch_size, seq_len, d_model]
        residual = inputs
        output = nn.LayerNorm(self.d_in).to('cuda:0')(inputs)
        output = self.fc(output)
        return nn.LayerNorm(self.d_in).to('cuda:0')(residual)+output


class inter_molecule_attention(nn.Module):
    def __init__(self, d_model, n_heads, d_k):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.W_Q = nn.Conv1d(d_model, d_k*n_heads, kernel_size=1, stride=1, padding=0)
        self.W_K = nn.Conv1d(d_model, d_k*n_heads, kernel_size=1, stride=1, padding=0)
        self.W_V = nn.Conv1d(d_model, d_k*n_heads, kernel_size=1, stride=1, padding=0)
        self.ScaledDotProductAttention = ScaledDotProductAttention(d_k)
        self.FFN = PoswiseFeedForwardNet(d_model, d_model*2)
        self.fc = nn.Linear(n_heads*d_k, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model*2, self.d_model, bias=False)
        )

    def forward(self, input_Q, input_K, input_V):
        residual, batch_size = input_Q, input_Q.size(0)
        '''residual, batch_size = input_Q, input_Q.size(0)'''
        #(B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q.transpose(1, 2)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #Q:[bs, heads, len_q, d_k]
        K = self.W_K(input_K.transpose(1, 2)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #K:[bs, heads, len_k, d_k]
        V = self.W_V(input_V.transpose(1, 2)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) #V:[bs, heads, len_v(=len_k), d_v]
        #print(Q.shape, K.shape, V.shape)
        #context:[batch_size, n_heads, len_q, d_v]
        #attn:[batch_size, n_heads, len_q, len_k]
        context = self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads*self.d_k) #context:[bs, len_q, heads*d_v]
        output = self.fc(context) #[batch_size, len_q, d_model]
        
        residual = self.W_Q(residual.transpose(1, 2)).view(-1, self.d_model)
        residual = nn.LayerNorm(self.d_model).to('cuda:0')(residual)
        residual = torch.relu(residual)
        output = residual+output

        residual = output
        output = nn.LayerNorm(self.d_model).to('cuda:0')(output)
        output = self.fc(output)
        
        return nn.LayerNorm(self.d_model).to('cuda:0')(residual)+output
    

class hierarchical_mutual_attention(nn.Module):
    def __init__(self, d_model, n_heads, d_k):
        super().__init__()
        self.HMCA = inter_molecule_attention(d_model, n_heads, d_k)
    def forward(self, f_x_decoded, f_y_decoded):
        mhca_1 = self.HMCA(f_x_decoded, f_y_decoded, f_y_decoded)
        mhca_2 = self.HMCA(f_y_decoded, mhca_1, mhca_1)
        hma_output = self.HMCA(mhca_1, mhca_2, mhca_2)
        return hma_output


class AHGRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.intra_enc = intra_domain_encoder(256, 4)
        self.inter_enc = inter_domain_encoder(256, 4)
        self.decoder = cross_domain_decoder(256, 4, 64)
        self.fusion = hierarchical_mutual_attention(512, 4, 128)

    def forward(self, n_x, n_y, data):
        torch.manual_seed(1)
        x_d = data['X']['attribute'].shape[1]
        f_x = nn.Linear(x_d, 256)(data['X']['attribute'])
        f_x_intra = self.intra_enc('X', f_x, data)
        y_d = data['Y']['attribute'].shape[1]
        f_y = nn.Linear(y_d, 256)(data['Y']['attribute'])
        f_y_intra = self.intra_enc('Y', f_y, data)

        f_x = nn.Linear(x_d, 256)(data['X']['attribute'])
        f_y = nn.Linear(y_d, 256)(data['Y']['attribute'])
        f_all = torch.cat([f_x, f_y])
        f_inter = self.inter_enc(f_all, data)
        f_x_inter, f_y_inter = torch.split(f_inter, (n_x, n_y), dim=1)
        f_x_inter = f_x_inter.view(1, n_x, -1)
        f_y_inter = f_y_inter.view(1, n_y, -1)

        f_x_decoded = self.decoder(f_x_intra, f_x_inter)
        f_y_decoded = self.decoder(f_y_intra, f_y_inter)

        hma1_output = self.fusion(f_x_decoded, f_y_decoded)
        hma2_output = self.fusion(f_y_decoded, f_x_decoded)

        mat_inter = hma1_output.view(-1, n_x).t().mm(hma2_output.view(-1, n_y))

        f_x = torch.cat([f_x_intra, f_x_inter], dim=-1)
        f_x = f_x.view(-1, n_x).t()
        f_y = torch.cat([f_y_intra, f_y_inter], dim=-1)
        f_y = f_y.view(-1, n_y).t()
        
        return f_x, f_y, mat_inter
    
    
def feature_representation(model,n_x, n_y, dataset):
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    model = train(model,n_x,n_y, dataset, optimizer)
    model.eval()
    with torch.no_grad():
        pept_feat, prot_feat, _ = model(n_x, n_y, dataset)

    return pept_feat.cpu().detach().numpy(), prot_feat.cpu().detach().numpy()


def train(model,n_x,n_y, train_data, optimizer):
    model.train()

    for epoch in range(0, 100):
        model.zero_grad()
        _, _, score = model(n_x, n_y, train_data)
        loss_cal = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss_cal(score, train_data['A'].cuda())
        loss.backward()
        optimizer.step()
    return model