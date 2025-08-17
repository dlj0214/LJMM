import torch
import torch.nn as nn


class LJMM(nn.Module):
    def __init__(self, input_audio_dim, input_video_dim, hidden_dim, latent_dim):
        super(LJMM, self).__init__()

        # 音频卷积特征提取
        self.audio_conv = nn.Conv1d(input_audio_dim, hidden_dim, kernel_size=1)

        # 视频卷积特征提取
        self.video_conv = nn.Conv1d(input_video_dim, hidden_dim, kernel_size=1)

        # 音频和视频到共享潜在空间的映射
        self.audio_latent = nn.Linear(hidden_dim, latent_dim)
        self.video_latent = nn.Linear(hidden_dim, latent_dim)

        # 多模态协作状态空间建模（CoSSM）
        self.shared_latent = nn.Linear(latent_dim * 2, latent_dim)  # 融合音频和视频特征

        # 多模态增强状态空间建模（EnSSM）
        self.res_block = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1)
        )

        # Bi-Mamba 结构
        self.bi_mamba = BiMamba(latent_dim)

        # 分类层
        self.fc = nn.Linear(latent_dim, 2)  # 二分类：抑郁症 vs 非抑郁症

    def forward(self, audio, video):
        # 提取音频和视频特征
        audio_features = self.audio_conv(audio)
        video_features = self.video_conv(video)

        # 映射到潜在空间
        audio_latent = self.audio_latent(audio_features)
        video_latent = self.video_latent(video_features)

        # 融合音频和视频潜在空间
        combined_latent = torch.cat((audio_latent, video_latent), dim=1)

        # 共享潜在空间
        shared_latent = self.shared_latent(combined_latent)

        # 通过残差块进行多模态增强
        enhanced_latent = self.res_block(shared_latent)

        # 通过Bi-Mamba建模上下文
        final_latent = self.bi_mamba(enhanced_latent)

        # 分类
        output = self.fc(final_latent)
        return output


class BiMamba(nn.Module):
    def __init__(self, input_dim):
        super(BiMamba, self).__init__()
        self.hidden_dim = input_dim
        self.state_matrix = nn.Parameter(torch.randn(input_dim, input_dim))
        self.input_matrix = nn.Parameter(torch.randn(input_dim, input_dim))
        self.output_matrix = nn.Parameter(torch.randn(input_dim, 1))

    def forward(self, x):
        # 实现Bi-Mamba的状态空间方程
        h = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        for t in range(x.size(1)):
            h = torch.matmul(h, self.state_matrix) + torch.matmul(x[:, t], self.input_matrix)
            output = torch.matmul(h, self.output_matrix)
        return output
