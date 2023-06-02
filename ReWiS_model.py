import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchinfo import summary
import torchsummary 

class ReWiS_LeNet(nn.Module):
    def __init__(self):
        super(ReWiS_LeNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=(3, 1)),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(96 * 3 * 13, 24),
            nn.ReLU(),
            nn.Linear(24, 4),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ReWiS_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=4):
        super(ReWiS_ResNet, self).__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(1, 3, 7, stride=(3, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, kernel_size=(10, 11), stride=1),
            nn.ReLU()
        )
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ReWiS_ResNet50():
    return ReWiS_ResNet(Bottleneck, [3, 4, 6, 3])

class ReWiS_RNN(nn.Module):
    def __init__(self,hidden_dim=64):
        super(ReWiS_RNN,self).__init__()
        self.rnn = nn.RNN(242,hidden_dim,num_layers=1)
        self.fc = nn.Linear(hidden_dim,4)
    def forward(self,x):
        x = x.view(-1,242,242)
        x = x.permute(1,0,2)
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        return outputs

class ReWiS_LSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ReWiS_LSTM, self).__init__()
        self.lstm = nn.LSTM(242, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        x = x.view(-1, 242, 242)
        x = x.permute(1, 0, 2)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class ReWiS_BiLSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ReWiS_BiLSTM, self).__init__()
        self.lstm = nn.LSTM(242, hidden_dim, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        x = x.view(-1, 242, 242)
        x = x.permute(1, 0, 2)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (C // self.num_heads)**0.5
        attn = attn.softmax(dim=-1)
        attn = self.att_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ReWiS_ViT(nn.Module):
    def __init__(self, in_channels, embed_dim, num_layers, num_heads, mlp_dim, num_classes, patch_size, in_size):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (in_size[0]*in_size[1]) // (patch_size[0] * patch_size[1]), embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.Sequential(
            *[ViTBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)

        x = self.norm(x[:, 0])
        x = self.fc(x)

        return x


if __name__ == "__main__" :
    model = ReWiS_ViT(
        in_channels=1,  # 입력 채널 수
        patch_size=[8, 64],  # 패치 크기 (세로, 가로) 242 = 2 * 11 * 11
        embed_dim=64,  # 임베딩 차원
        num_layers=12,  # Transformer 블록 수
        num_heads=8,  # 멀티헤드 어텐션에서의 헤드 수
        mlp_dim=4,  # MLP의 확장 비율
        num_classes=4,  # 분류할 클래스 수
        in_size=[64, 64]  # 입력 이미지 크기 (가로, 세로)
    ).to("cuda")

    input_size = (1,64,64)
    # print(summary(model, input_size = input_size))
    print(torchsummary.summary(model, input_size = input_size))


'''
class ReWiS_LeNet(nn.Module):
    def __init__(self):
        super(ReWiS_LeNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),  # output: [32, 60, 252]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: [32, 30, 126]
            nn.Conv2d(32, 64, kernel_size=5, stride=1),  # output: [64, 26, 122]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output: [64, 13, 61]
            nn.Conv2d(64, 96, kernel_size=3, stride=1),  # output: [96, 11, 59]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # output: [96, 5, 29]
        )
        
        self.fc = nn.Sequential(
            nn.Linear(96 * 5 * 29, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)

        return x

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ReWiS_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=4):
        super(ReWiS_ResNet, self).__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(1, 3, 7, stride=(3, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, kernel_size=(10, 11), stride=1),
            nn.ReLU()
        )
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

def ReWiS_ResNet50():
    return ReWiS_ResNet(Bottleneck, [3, 4, 6, 3])

class ReWiS_RNN(nn.Module):
    def __init__(self,hidden_dim=64):
        super(ReWiS_RNN,self).__init__()
        self.rnn = nn.RNN(256,hidden_dim,num_layers=1)
        self.fc = nn.Linear(hidden_dim,4)
    def forward(self,x):
        x = x.view(-1,64,256)
        x = x.permute(1,0,2)
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        return outputs

class ReWiS_LSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ReWiS_LSTM, self).__init__()
        self.lstm = nn.LSTM(256, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        x = x.view(-1, 64, 256)
        x = x.permute(1, 0, 2)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs

class ReWiS_BiLSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ReWiS_BiLSTM, self).__init__()
        self.lstm = nn.LSTM(256, hidden_dim, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        x = x.view(-1, 64, 256)
        x = x.permute(1, 0, 2)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (C // self.num_heads)**0.5
        attn = attn.softmax(dim=-1)
        attn = self.att_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ReWiS_ViT(nn.Module):
    def __init__(self, in_channels, embed_dim, num_layers, num_heads, mlp_dim, num_classes, patch_size, in_size):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (in_size[0]*in_size[1]) // (patch_size[0] * patch_size[1]), embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.Sequential(
            *[ViTBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)

        x = self.norm(x[:, 0])
        x = self.fc(x)

        return x
'''

'''
class ReWiS_LeNet(nn.Module):
    def __init__(self):
        super(ReWiS_LeNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 1), stride=(3, 1)),  # output: [32, 8, 28]
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=(2, 1)),  # output: [32, 4, 28]
            nn.Conv2d(32, 64, kernel_size=(5, 4), stride=(2, 2), padding=(2, 0)),  # output: [64, 2, 14]
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=(2, 1)),  # output: [64, 1, 14]
            nn.Conv2d(64, 96, kernel_size=(3, 3), stride=1, padding=1),  # output: [96, 1, 14]
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=(2, 1))  # output: [96, 1, 14]
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2976, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 4)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)

        return x

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ReWiS_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=4):
        super(ReWiS_ResNet, self).__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(1, 3, 7, stride=(3, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 3, kernel_size=(10, 11), stride=1),
            nn.ReLU()
        )
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

def ReWiS_ResNet50():
    return ReWiS_ResNet(Bottleneck, [3, 4, 6, 3])

class ReWiS_RNN(nn.Module):
    def __init__(self,hidden_dim=64):
        super(ReWiS_RNN,self).__init__()
        self.rnn = nn.RNN(64,hidden_dim,num_layers=1)
        self.fc = nn.Linear(hidden_dim,4)
    def forward(self,x):
        x = x.view(-1,64,64)
        x = x.permute(1,0,2)
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        return outputs

class ReWiS_LSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ReWiS_LSTM, self).__init__()
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        x = x.view(-1, 64, 64)
        x = x.permute(1, 0, 2)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs

class ReWiS_BiLSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ReWiS_BiLSTM, self).__init__()
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        x = x.view(-1, 64, 64)
        x = x.permute(1, 0, 2)
        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (C // self.num_heads)**0.5
        attn = attn.softmax(dim=-1)
        attn = self.att_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ReWiS_ViT(nn.Module):
    def __init__(self, in_channels, embed_dim, num_layers, num_heads, mlp_dim, num_classes, patch_size, in_size):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (in_size[0]*in_size[1]) // (patch_size[0] * patch_size[1]), embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.Sequential(
            *[ViTBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)

        x = self.norm(x[:, 0])
        x = self.fc(x)

        return x
'''