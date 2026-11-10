import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output,attention_weights
class CrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttention, self).__init__()
        
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads
        
        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by num_heads"
        
        # 查询、键、值的线性变换
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # self.dropout = nn.Dropout(0.)

        # self.norm = nn.LayerNorm(512)
        # self.act = nn.ReLU()
        # # 最终输出的线性变换
        self.fc_out = nn.Linear(embed_size, embed_size,bias=False)
    def forward(self, query_input, key_input, value_input, mask=None):
        batch_size = query_input.shape[0]
        
        # 1. 将输入（Q, K, V）进行线性变换
        Q = self.query(query_input)  # 查询来自 query_input
        K = self.key(key_input)      # 键来自 key_input
        V = self.value(value_input)  # 值来自 value_input
        
        # 2. 将 Q, K, V 分割成多个头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 计算 Q 和 K 的点积，得到能量（Attention Score）
        energy = torch.matmul(Q, K.transpose(-1, -2))  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # 缩放
        scale = self.head_dim ** 0.5
        energy = energy / scale
        
        # 4. 如果提供了mask，应用mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        # 5. 使用 softmax 归一化，得到注意力权重
        attention_weights = torch.softmax(energy, dim=-1) # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # 6. 将注意力权重应用到 V 上，得到最终的注意力输出
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len_q, head_dim)
        
        # 7. 将多个头的结果拼接起来
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # 8. 输出线性变换
        # out = self.act(self.norm(attention_output))
        out = self.fc_out(attention_output)
        
        return out, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        # 每个头的维度大小
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads
        
        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by num_heads"
        
        # 查询、键、值的线性变换
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # 最终输出的线性变换
        # self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, value, key, query, mask=None):
        batch_size = query.shape[0]
        
        # 1. 将输入（Q, K, V）进行线性变换
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # 2. 将 Q, K, V 分割成多个头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 将多个头的结果拼接起来
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # 5. 输出线性变换
        # out = self.fc_out(attention_output)
        
        return attention_output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask):
        # Q: (batch_size, num_heads, seq_len, head_dim)
        # K: (batch_size, num_heads, seq_len, head_dim)
        # V: (batch_size, num_heads, seq_len, head_dim)
        
        # 计算 Q 和 K 的点积
        energy = torch.matmul(Q, K.transpose(-1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        
        # 缩放
        scale = self.head_dim ** 0.5
        energy = energy / scale
        
        # 如果提供了mask，应用mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        # 使用softmax归一化，得到注意力权重
        attention_weights = torch.softmax(energy, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        # 将注意力权重应用到V上，得到最终的注意力输出
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        return attention_output, attention_weights
    
class SingleHeadCrossAttention(nn.Module):
    def __init__(self, embed_size):
        """
        单头交叉注意力模块
        :param embed_size: 输入特征的维度 (d_model)
        """
        super(SingleHeadCrossAttention, self).__init__()
        self.embed_size = embed_size
        
        # 查询 (Query)、键 (Key)、值 (Value) 的线性变换
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # 输出的线性变换
        self.out = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        """
        前向传播
        :param x: 输入1 (Query), shape: (batch_size, seq_len1, embed_size)
        :param y: 输入2 (Key/Value), shape: (batch_size, seq_len2, embed_size)
        :return: 输出交叉注意力特征, shape: (batch_size, seq_len1, embed_size)
        """
        # 线性变换得到 Q, K, V
        Q = self.query(x)  # (batch_size, seq_len1, embed_size)
        K = self.key(y)    # (batch_size, seq_len2, embed_size)
        V = self.value(y)  # (batch_size, seq_len2, embed_size)
        
        # 计算注意力分数
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_len1, seq_len2)
        attention_scores = attention_scores / (self.embed_size ** 0.5)  # 缩放
        attention_weights = self.softmax(attention_scores)  # (batch_size, seq_len1, seq_len2)
        
        # 加权求和得到输出
        attention_output = torch.bmm(attention_weights, V)  # (batch_size, seq_len1, embed_size)
        output = self.out(attention_output)  # 线性变换 (batch_size, seq_len1, embed_size)
        
        return output, attention_weights
    
class CanAttention(nn.Module):
    def __init__(self,num_class=5):
        super(CanAttention, self).__init__()
      
        self.trad_linear1 = nn.Linear(100,128,bias=False)
        self.trad_bn1 = nn.BatchNorm1d(128)
        self.trad_act1 = nn.ReLU()

        self.trad_linear2 = nn.Linear(128,256,bias=False)
        self.trad_bn2 = nn.BatchNorm1d(256)
        self.trad_act2 = nn.ReLU()

        self.trad_linear3 = nn.Linear(256,512,bias=False)
        self.trad_bn3 = nn.BatchNorm1d(512)
        self.trad_act3 = nn.ReLU()

        self.linear1 = nn.Sequential(
                                    nn.Linear(512*3,1024,bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(),
                                    nn.Linear(1024,2048),
                                    # nn.BatchNorm1d(1024),
                                    # nn.ReLU(),
                                    )
        self.linear2 = nn.Linear(512*3,512,bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.ReLU()
        self.classhead = nn.Linear(512,num_class)
        

    
    def forward(self, x, y, v):
        # print("image shape:",x.shape)
        # print("point shape:",y.shape)
        # print("trad shape:",v.shape)
        b,d = x.shape
        # v = torch.tensor(v).to(dtype=torch.float32,device='cuda:0')
        v = self.trad_act1(self.trad_bn1(self.trad_linear1(v)))
        v = self.trad_act2(self.trad_bn2(self.trad_linear2(v)))
        v = self.trad_act3(self.trad_bn3(self.trad_linear3(v)))

        z = torch.cat((x,y,v),dim=1)
        features = self.linear1(z)

        z = self.act1(self.bn1(self.linear2(z)))
        logits = self.classhead(z)
        return z, features, logits
    
class CrossAttentionFusion(nn.Module):
    def __init__(self, featuresize=256,numclass=5):
        super(CrossAttentionFusion,self).__init__()
        self.trad_linear2 = nn.Linear(100,128,bias=False)
        self.trad_bn2 = nn.BatchNorm1d(128)
        self.trad_act2 = nn.ReLU()

        self.trad_linear3 = nn.Linear(128,256,bias=False)
        self.trad_bn3 = nn.BatchNorm1d(256)
        self.trad_act3 = nn.ReLU()

        self.trad_linear4 = nn.Linear(256,512,bias=False)
        self.trad_bn4 = nn.BatchNorm1d(512)
        self.trad_act4 = nn.ReLU()

        # self.crossMha = SingleHeadCrossAttention(embed_size=512)
        self.crossMha = CrossAttention(embed_size=512, num_heads = 16)
        # self.drop1 = nn.Dropout(0.)
        self.norm1 = nn.LayerNorm(512)

        self.ffn = nn.Sequential(
            nn.Linear(512,2048,bias=False),
            nn.ReLU(),
            nn.Linear(2048,512,bias=False),
        )
        self.drop2 = nn.Dropout(0.)
        self.norm2 = nn.LayerNorm(512)
        self.act = nn.ReLU()

        # self.classhead1 = nn.Linear(512*2,512,bias=False)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.act1 = nn.ReLU(negative_slope=0.2)

        self.classhead2 = nn.Linear(512,5)

        

    def forward(self,x,y,v):
        '''
        x->16*512 //image_feature
        y->16*512 //point_feature -->>32*16*512
        x:传统特征
        '''
        b,_ = x.size()
        v = torch.tensor(v).to(dtype=torch.float32,device='cuda:0')
        v = self.trad_act2(self.trad_bn2(self.trad_linear2(v)))
        v = self.trad_act3(self.trad_bn3(self.trad_linear3(v)))
        v = self.trad_act4(self.trad_bn4(self.trad_linear4(v)))
        z = torch.stack((x,v),dim=1) 
        z = torch.relu(self.norm1(self.crossMha(y,z,z)[0]))+y
        z = self.act(self.drop2(self.norm2(self.ffn(z))))+z
        z = z.mean(dim=1)
        # z = torch.cat((z.mean(1),z.max(dim=1)[0]),dim=1)
        # z = torch.cat((x,y,v,z),dim=1)
        # z = self.act1(self.bn1(self.classhead1(z))) 
        z = self.classhead2(z)

        return z

    
