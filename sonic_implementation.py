"""
Sonic: Shifting Focus to Global Audio Perception in Portrait Animation
基于论文和推理代码的完整实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import cv2
from dataclasses import dataclass
from einops import rearrange, repeat
import math

# ==================== 配置类 ====================
@dataclass
class SonicConfig:
    """Sonic模型配置"""
    # 模型参数
    num_frames: int = 25
    resolution: int = 224
    audio_context_frames: int = 5  # 0.2s音频上下文
    
    # 扩散模型参数
    num_inference_steps: int = 25
    guidance_scale_image: float = 3.0
    guidance_scale_audio: float = 2.5
    
    # 运动控制参数
    motion_bucket_translation: int = 127  # 头部运动幅度
    motion_bucket_expression: int = 127   # 表情运动幅度
    dynamic_scale: float = 1.5  # 1.0=mild, 1.5=moderate, 2.0=intense
    
    # 时间感知位置偏移融合参数
    shift_offset: int = 3
    
    # 音频特征参数
    audio_feature_dim: int = 384  # Whisper-Tiny特征维度
    audio_proj_dim: int = 768
    
    # UNet参数
    unet_in_channels: int = 8
    unet_out_channels: int = 4
    cross_attention_dim: int = 320

# ==================== 音频处理模块 ====================
class AudioEncoder(nn.Module):
    """音频编码器 - 使用Whisper-Tiny提取特征"""
    def __init__(self, config: SonicConfig):
        super().__init__()
        self.config = config
        
        # 音频特征投影层
        self.audio_proj = nn.Sequential(
            nn.Linear(config.audio_feature_dim, config.audio_proj_dim),
            nn.ReLU(),
            nn.Linear(config.audio_proj_dim, config.audio_proj_dim),
            nn.ReLU(),
            nn.Linear(config.audio_proj_dim, config.cross_attention_dim)
        )
        
        # 时间维度池化用于时间音频注意力
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_features: [B, T, D] Whisper提取的音频特征
        Returns:
            dict: 包含空间和时间音频嵌入
        """
        B, T, D = audio_features.shape
        
        # 投影音频特征
        audio_embed = self.audio_proj(audio_features)  # [B, T, cross_attention_dim]
        
        # 空间音频嵌入 - 保持完整时间维度
        spatial_audio_embed = audio_embed
        
        # 时间音频嵌入 - 池化时间维度
        temporal_audio_embed = self.temporal_pool(audio_embed.transpose(1, 2)).transpose(1, 2)
        temporal_audio_embed = repeat(temporal_audio_embed, 'b 1 d -> b t d', t=T)
        
        return {
            'spatial': spatial_audio_embed,
            'temporal': temporal_audio_embed
        }

# ==================== 上下文增强音频学习模块 ====================
# class ContextEnhancedAudioLearning(nn.Module):
#     """上下文增强音频学习 - 论文核心创新之一"""
#     def __init__(self, config: SonicConfig):
#         super().__init__()
#         self.config = config
        
#         # 空间音频交叉注意力
#         self.spatial_audio_attn = nn.MultiheadAttention(
#             embed_dim=config.cross_attention_dim,
#             num_heads=8,
#             batch_first=True
#         )
        
#         # 时间音频交叉注意力
#         self.temporal_audio_attn = nn.MultiheadAttention(
#             embed_dim=config.cross_attention_dim,
#             num_heads=8,
#             batch_first=True
#         )
        
#         # 层归一化
#         self.norm1 = nn.LayerNorm(config.cross_attention_dim)
#         self.norm2 = nn.LayerNorm(config.cross_attention_dim)
        
#     def forward(self, 
#                 spatial_features: torch.Tensor,
#                 temporal_features: torch.Tensor,
#                 audio_embeddings: Dict[str, torch.Tensor],
#                 face_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         执行上下文增强的音频学习
        
#         Args:
#             spatial_features: [B, H*W, D] 空间特征
#             temporal_features: [B, T, D] 时间特征
#             audio_embeddings: 音频嵌入字典
#             face_mask: [B, H, W] 面部区域掩码
#         """
#         # 空间音频注意力 - 等式(1)
#         if face_mask is not None:
#             # 限制在面部区域
#             mask_flat = rearrange(face_mask, 'b h w -> b (h w)')
#             spatial_features_masked = spatial_features * mask_flat.unsqueeze(-1)
#         else:
#             spatial_features_masked = spatial_features
            
#         spatial_out, _ = self.spatial_audio_attn(
#             query=spatial_features_masked,
#             key=audio_embeddings['spatial'],
#             value=audio_embeddings['spatial']
#         )
#         spatial_out = self.norm1(spatial_features + spatial_out)
        
#         # 时间音频注意力 - 等式(2)
#         temporal_out, _ = self.temporal_audio_attn(
#             query=temporal_features,
#             key=audio_embeddings['temporal'],
#             value=audio_embeddings['temporal']
#         )
#         temporal_out = self.norm2(temporal_features + temporal_out)
        
#         return spatial_out, temporal_out

# class ContextEnhancedAudioLearning(nn.Module):
#     """上下文增强音频学习 - 论文核心创新之一"""
#     def __init__(self, config: SonicConfig):
#         super().__init__()
#         self.config = config
        
#         # 空间音频交叉注意力 - 修改嵌入维度为320
#         self.spatial_audio_attn = nn.MultiheadAttention(
#             embed_dim=config.cross_attention_dim,  # 320
#             num_heads=8,
#             batch_first=True
#         )
        
#         # 时间音频交叉注意力
#         self.temporal_audio_attn = nn.MultiheadAttention(
#             embed_dim=config.cross_attention_dim,  # 320
#             num_heads=8,
#             batch_first=True
#         )
        
#         # 层归一化
#         self.norm1 = nn.LayerNorm(config.cross_attention_dim)
#         self.norm2 = nn.LayerNorm(config.cross_attention_dim)
        
#     def forward(self, 
#                 spatial_features: torch.Tensor,
#                 temporal_features: torch.Tensor,
#                 audio_embeddings: Dict[str, torch.Tensor],
#                 face_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         执行上下文增强的音频学习
#         """
#         # 确保音频特征与空间特征的序列长度匹配
#         spatial_seq_len = spatial_features.shape[1]
#         audio_spatial = audio_embeddings['spatial']
        
#         # 调整音频特征序列长度以匹配空间特征
#         if audio_spatial.shape[1] != spatial_seq_len:
#             if audio_spatial.shape[1] > spatial_seq_len:
#                 # 截断
#                 audio_spatial = audio_spatial[:, :spatial_seq_len, :]
#             else:
#                 # 填充或重复
#                 repeat_factor = (spatial_seq_len // audio_spatial.shape[1]) + 1
#                 audio_spatial = audio_spatial.repeat(1, repeat_factor, 1)
#                 audio_spatial = audio_spatial[:, :spatial_seq_len, :]
        
#         # 确保时间特征与音频时间特征匹配
#         temporal_seq_len = temporal_features.shape[1]
#         audio_temporal = audio_embeddings['temporal']
        
#         if audio_temporal.shape[1] != temporal_seq_len:
#             if audio_temporal.shape[1] > temporal_seq_len:
#                 audio_temporal = audio_temporal[:, :temporal_seq_len, :]
#             else:
#                 repeat_factor = (temporal_seq_len // audio_temporal.shape[1]) + 1
#                 audio_temporal = audio_temporal.repeat(1, repeat_factor, 1)
#                 audio_temporal = audio_temporal[:, :temporal_seq_len, :]
        
#         # 空间音频注意力
#         if face_mask is not None:
#             mask_flat = rearrange(face_mask, 'b h w -> b (h w)')
#             spatial_features_masked = spatial_features * mask_flat.unsqueeze(-1)
#         else:
#             spatial_features_masked = spatial_features
            
#         spatial_out, _ = self.spatial_audio_attn(
#             query=spatial_features_masked,
#             key=audio_spatial,
#             value=audio_spatial
#         )
#         spatial_out = self.norm1(spatial_features + spatial_out)
        
#         # 时间音频注意力
#         temporal_out, _ = self.temporal_audio_attn(
#             query=temporal_features,
#             key=audio_temporal,
#             value=audio_temporal
#         )
#         temporal_out = self.norm2(temporal_features + temporal_out)
        
#         return spatial_out, temporal_out

class ContextEnhancedAudioLearning(nn.Module):
    """上下文增强音频学习 - 论文核心创新之一"""
    def __init__(self, config: SonicConfig):
        super().__init__()
        self.config = config
        
        # 空间音频交叉注意力
        self.spatial_audio_attn = nn.MultiheadAttention(
            embed_dim=config.cross_attention_dim,  # 320
            num_heads=8,
            batch_first=True
        )
        
        # 时间音频交叉注意力
        self.temporal_audio_attn = nn.MultiheadAttention(
            embed_dim=config.cross_attention_dim,  # 320
            num_heads=8,
            batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(config.cross_attention_dim)
        self.norm2 = nn.LayerNorm(config.cross_attention_dim)
        
    def forward(self, 
                spatial_features: torch.Tensor,
                temporal_features: torch.Tensor,
                audio_embeddings: Dict[str, torch.Tensor],
                face_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行上下文增强的音频学习
        
        Args:
            spatial_features: [B*T, H*W, D] 空间特征
            temporal_features: [B, T, D] 时间特征
            audio_embeddings: 音频嵌入字典
            face_mask: [B, H, W] 面部区域掩码
        """
        # 打印调试信息
        print(f"spatial_features shape: {spatial_features.shape}")
        print(f"temporal_features shape: {temporal_features.shape}")
        print(f"audio_embeddings['spatial'] shape: {audio_embeddings['spatial'].shape}")
        print(f"audio_embeddings['temporal'] shape: {audio_embeddings['temporal'].shape}")
        
        # 获取音频特征
        audio_spatial = audio_embeddings['spatial']  # [B, T_audio, D]
        audio_temporal = audio_embeddings['temporal']  # [B, T_audio, D]
        
        # 处理空间音频注意力
        B_times_T, spatial_seq_len, spatial_dim = spatial_features.shape
        B_audio, T_audio, audio_dim = audio_spatial.shape
        
        # 确保维度匹配
        if spatial_dim != self.config.cross_attention_dim:
            raise ValueError(f"Spatial features dim {spatial_dim} != expected {self.config.cross_attention_dim}")
        if audio_dim != self.config.cross_attention_dim:
            raise ValueError(f"Audio features dim {audio_dim} != expected {self.config.cross_attention_dim}")
            
        # 调整音频特征以匹配空间特征的批次大小
        if B_audio != B_times_T:
            # 假设 B_times_T = B * T，我们需要重复音频特征
            T_video = B_times_T // B_audio
            audio_spatial_expanded = audio_spatial.unsqueeze(1).repeat(1, T_video, 1, 1)  # [B, T_video, T_audio, D]
            audio_spatial_expanded = audio_spatial_expanded.reshape(B_times_T, T_audio, audio_dim)  # [B*T, T_audio, D]
        else:
            audio_spatial_expanded = audio_spatial
            
        # 调整音频序列长度以匹配空间序列长度（如果需要）
        if T_audio != spatial_seq_len:
            if T_audio > spatial_seq_len:
                # 截断音频特征
                audio_spatial_final = audio_spatial_expanded[:, :spatial_seq_len, :]
            else:
                # 填充或重复音频特征
                repeat_factor = (spatial_seq_len + T_audio - 1) // T_audio  # 向上取整
                audio_spatial_repeated = audio_spatial_expanded.repeat(1, repeat_factor, 1)
                audio_spatial_final = audio_spatial_repeated[:, :spatial_seq_len, :]
        else:
            audio_spatial_final = audio_spatial_expanded
            
        print(f"After adjustment - audio_spatial_final shape: {audio_spatial_final.shape}")
        print(f"spatial_features final shape: {spatial_features.shape}")
        
        # 应用面部掩码（如果提供）
        if face_mask is not None:
            # 这里需要调整掩码的处理方式
            # 由于spatial_features是[B*T, H*W, D]，我们需要相应调整掩码
            pass  # 暂时跳过面部掩码处理
            
        spatial_features_input = spatial_features
        
        # 空间音频注意力
        try:
            spatial_out, _ = self.spatial_audio_attn(
                query=spatial_features_input,
                key=audio_spatial_final,
                value=audio_spatial_final
            )
            spatial_out = self.norm1(spatial_features + spatial_out)
        except Exception as e:
            print(f"Spatial attention error: {e}")
            print(f"Query shape: {spatial_features_input.shape}")
            print(f"Key/Value shape: {audio_spatial_final.shape}")
            raise e
        
        # 处理时间音频注意力
        B_temp, T_temp, temp_dim = temporal_features.shape
        
        # 调整时间音频特征
        if T_audio != T_temp:
            if T_audio > T_temp:
                audio_temporal_final = audio_temporal[:, :T_temp, :]
            else:
                repeat_factor = (T_temp + T_audio - 1) // T_audio
                audio_temporal_repeated = audio_temporal.repeat(1, repeat_factor, 1)
                audio_temporal_final = audio_temporal_repeated[:, :T_temp, :]
        else:
            audio_temporal_final = audio_temporal
            
        print(f"temporal_features shape: {temporal_features.shape}")  
        print(f"audio_temporal_final shape: {audio_temporal_final.shape}")
        
        # 时间音频注意力
        try:
            temporal_out, _ = self.temporal_audio_attn(
                query=temporal_features,
                key=audio_temporal_final,
                value=audio_temporal_final
            )
            temporal_out = self.norm2(temporal_features + temporal_out)
        except Exception as e:
            print(f"Temporal attention error: {e}")
            print(f"Query shape: {temporal_features.shape}")
            print(f"Key/Value shape: {audio_temporal_final.shape}")
            raise e
        
        return spatial_out, temporal_out

# ==================== 运动解耦控制器 ====================
class MotionDecoupledController(nn.Module):
    """运动解耦控制器 - 独立控制头部和表情运动"""
    def __init__(self, config: SonicConfig):
        super().__init__()
        self.config = config
        
        # 音频到运动桶的映射网络
        self.audio2bucket = nn.Sequential(
            nn.Linear(config.cross_attention_dim + 768, 512),  # audio + clip特征
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 输出translation和expression两个值
        )
        
        # 运动桶嵌入
        self.motion_embed_dim = 256
        self.translation_embed = nn.Embedding(256, self.motion_embed_dim)
        self.expression_embed = nn.Embedding(256, self.motion_embed_dim)
        
        # 位置编码
        self.pos_encoding = self._create_sinusoidal_embedding(256, self.motion_embed_dim)
        
    def _create_sinusoidal_embedding(self, num_positions: int, dim: int) -> torch.Tensor:
        """创建正弦位置编码"""
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        
        pe = torch.zeros(num_positions, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
        
    def forward(self, 
                audio_features: torch.Tensor,
                clip_features: torch.Tensor,
                manual_buckets: Optional[Dict[str, int]] = None) -> Dict[str, torch.Tensor]:
        """
        计算运动控制嵌入
        
        Args:
            audio_features: [B, D] 池化后的音频特征
            clip_features: [B, D] CLIP图像特征
            manual_buckets: 手动指定的运动桶值
        """
        if manual_buckets is None:
            # 自动预测运动桶 - 等式(4)
            combined_features = torch.cat([audio_features, clip_features], dim=-1)
            predicted_buckets = self.audio2bucket(combined_features)  # [B, 2]
            
            # 应用动态缩放
            predicted_buckets = predicted_buckets * self.config.dynamic_scale
            
            # 限制在有效范围内
            translation_bucket = torch.clamp(predicted_buckets[:, 0], 0, 255).long()
            expression_bucket = torch.clamp(predicted_buckets[:, 1], 0, 255).long()
        else:
            translation_bucket = torch.tensor([manual_buckets['translation']], device=audio_features.device)
            expression_bucket = torch.tensor([manual_buckets['expression']], device=audio_features.device)
            
        # 获取运动嵌入 - 等式(3)
        translation_embed = self.translation_embed(translation_bucket)
        expression_embed = self.expression_embed(expression_bucket)
        
        # 添加位置编码
        pos_encoding = self.pos_encoding.to(audio_features.device)
        translation_embed = translation_embed + pos_encoding[translation_bucket]
        expression_embed = expression_embed + pos_encoding[expression_bucket]
        
        return {
            'translation': translation_embed,
            'expression': expression_embed,
            'combined': translation_embed + expression_embed,
            'buckets': {
                'translation': translation_bucket,
                'expression': expression_bucket
            }
        }

# ==================== 时间感知位置偏移融合 ====================
class TimeAwarePositionShiftFusion(nn.Module):
    """时间感知位置偏移融合 - 用于长视频生成的核心创新"""
    def __init__(self, config: SonicConfig):
        super().__init__()
        self.config = config
        self.shift_offset = config.shift_offset
        
    def forward(self, 
                latents: torch.Tensor,
                audio_features: torch.Tensor,
                timestep: int,
                total_timesteps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行时间感知的位置偏移融合
        
        Args:
            latents: [B, C, T, H, W] 视频潜在表示
            audio_features: [B, T_audio, D] 音频特征
            timestep: 当前扩散时间步
            total_timesteps: 总时间步数
        """
        B, C, T, H, W = latents.shape
        T_audio = audio_features.shape[1]
        
        # 计算累积偏移 - 算法1
        cumulative_offset = (total_timesteps - timestep) * self.shift_offset
        cumulative_offset = cumulative_offset % T  # 循环填充
        
        # 对潜在表示进行偏移
        if cumulative_offset > 0:
            # 循环偏移潜在表示
            latents_shifted = torch.cat([
                latents[:, :, -cumulative_offset:],
                latents[:, :, :-cumulative_offset]
            ], dim=2)
            
            # 对音频特征进行相应偏移
            audio_offset = cumulative_offset * (T_audio // T)
            audio_shifted = torch.cat([
                audio_features[:, -audio_offset:],
                audio_features[:, :-audio_offset]
            ], dim=1)
        else:
            latents_shifted = latents
            audio_shifted = audio_features
            
        return latents_shifted, audio_shifted

# ==================== Sonic UNet ====================
class SonicUNet(nn.Module):
    """Sonic UNet - 集成所有模块的主网络"""
    def __init__(self, config: SonicConfig):
        super().__init__()
        self.config = config
        
        # 核心模块
        self.audio_encoder = AudioEncoder(config)
        self.context_audio_learning = ContextEnhancedAudioLearning(config)
        self.motion_controller = MotionDecoupledController(config)
        self.shift_fusion = TimeAwarePositionShiftFusion(config)
        
        # 简化的UNet结构（实际实现中应该使用完整的SVD UNet）
        self.time_embed = nn.Sequential(
            nn.Linear(256, config.cross_attention_dim),
            nn.SiLU(),
            nn.Linear(config.cross_attention_dim, config.cross_attention_dim)
        )
        
        # 输入投影
        self.conv_in = nn.Conv3d(
            config.unet_in_channels, 
            320, 
            kernel_size=3, 
            padding=1
        )
        
        # 输出投影
        self.conv_out = nn.Conv3d(
            320,
            config.unet_out_channels,
            kernel_size=3,
            padding=1
        )
        
    # def forward(self,
    #             sample: torch.Tensor,
    #             timestep: torch.Tensor,
    #             encoder_hidden_states: Dict[str, torch.Tensor],
    #             audio_features: torch.Tensor,
    #             clip_features: torch.Tensor,
    #             face_mask: Optional[torch.Tensor] = None,
    #             manual_motion_buckets: Optional[Dict[str, int]] = None) -> torch.Tensor:
    #     """
    #     前向传播
        
    #     Args:
    #         sample: [B, C, T, H, W] 噪声潜在表示
    #         timestep: [B] 时间步
    #         encoder_hidden_states: 编码器隐藏状态
    #         audio_features: 音频特征
    #         clip_features: CLIP特征
    #         face_mask: 面部掩码
    #         manual_motion_buckets: 手动运动控制
    #     """
    #     B, C, T, H, W = sample.shape
        
    #     # 1. 时间嵌入
    #     t_emb = self.get_timestep_embedding(timestep, 256)
    #     t_emb = self.time_embed(t_emb)
        
    #     # 2. 音频编码
    #     audio_embeddings = self.audio_encoder(audio_features)
        
    #     # 3. 运动控制
    #     audio_pooled = audio_embeddings['spatial'].mean(dim=1)  # 池化用于运动预测
    #     motion_embeddings = self.motion_controller(
    #         audio_pooled, 
    #         clip_features,
    #         manual_motion_buckets
    #     )
        
    #     # 4. 时间感知位置偏移（如果处于推理阶段）
    #     if not self.training:
    #         sample, audio_features_shifted = self.shift_fusion(
    #             sample, 
    #             audio_features,
    #             timestep[0].item(),
    #             self.config.num_inference_steps
    #         )
    #         # 重新编码偏移后的音频
    #         audio_embeddings = self.audio_encoder(audio_features_shifted)
        
    #     # 5. UNet前向传播（简化版本）
    #     hidden_states = self.conv_in(sample)
        
    #     # 将3D特征重塑为2D进行注意力计算
    #     hidden_states_2d = rearrange(hidden_states, 'b c t h w -> (b t) c h w')
    #     spatial_features = rearrange(hidden_states_2d, 'bt c h w -> bt (h w) c')
    #     temporal_features = rearrange(hidden_states, 'b c t h w -> b t (c h w)')
        
    #     # 确保特征维度正确
    #     if spatial_features.shape[-1] != self.config.cross_attention_dim:
    #         # 添加投影层来匹配维度
    #         if not hasattr(self, 'spatial_proj'):
    #             self.spatial_proj = nn.Linear(spatial_features.shape[-1], self.config.cross_attention_dim).to(spatial_features.device)
    #         spatial_features = self.spatial_proj(spatial_features)
        
    #     if temporal_features.shape[-1] != self.config.cross_attention_dim:
    #         if not hasattr(self, 'temporal_proj'):
    #             self.temporal_proj = nn.Linear(temporal_features.shape[-1], self.config.cross_attention_dim).to(temporal_features.device)
    #         temporal_features = self.temporal_proj(temporal_features)
        
        
    #     # 6. 上下文增强音频学习
    #     spatial_out, temporal_out = self.context_audio_learning(
    #         spatial_features,
    #         temporal_features,
    #         audio_embeddings,
    #         face_mask
    #     )
        
    #     # 重塑回3D
    #     spatial_out = rearrange(spatial_out, '(b t) (h w) c -> b c t h w', 
    #                            b=B, t=T, h=H, w=W)
        
    #     # 7. 添加运动控制嵌入
    #     motion_embed = motion_embeddings['combined']
    #     motion_embed = repeat(motion_embed, 'b d -> b d t h w', t=T, h=H, w=W)
    #     hidden_states = spatial_out + motion_embed
        
    #     # 8. 输出投影
    #     output = self.conv_out(hidden_states)
        
    #     return output
    def forward(self,
                sample: torch.Tensor,
                timestep: torch.Tensor,
                encoder_hidden_states: Dict[str, torch.Tensor],
                audio_features: torch.Tensor,
                clip_features: torch.Tensor,
                face_mask: Optional[torch.Tensor] = None,
                manual_motion_buckets: Optional[Dict[str, int]] = None) -> torch.Tensor:
        """
        前向传播 - 完整修复版本
        """
        B, C, T, H, W = sample.shape
        print(f"Input sample shape: {sample.shape}")
        
        # 1. 时间嵌入
        t_emb = self.get_timestep_embedding(timestep, 256)
        t_emb = self.time_embed(t_emb)
        
        # 2. 音频编码
        print(f"Audio features input shape: {audio_features.shape}")
        audio_embeddings = self.audio_encoder(audio_features)
        print(f"Audio embeddings shapes - spatial: {audio_embeddings['spatial'].shape}, temporal: {audio_embeddings['temporal'].shape}")
        
        # 3. 运动控制
        audio_pooled = audio_embeddings['spatial'].mean(dim=1)  # [B, D]
        motion_embeddings = self.motion_controller(
            audio_pooled, 
            clip_features,
            manual_motion_buckets
        )
        
        # 4. 时间感知位置偏移（如果处于推理阶段）
        if not self.training:
            sample, audio_features_shifted = self.shift_fusion(
                sample, 
                audio_features,
                timestep[0].item(),
                self.config.num_inference_steps
            )
            # 重新编码偏移后的音频
            audio_embeddings = self.audio_encoder(audio_features_shifted)
        
        # 5. UNet前向传播
        hidden_states = self.conv_in(sample)  # [B, C_new, T, H_new, W_new]
        print(f"After conv_in shape: {hidden_states.shape}")
        
        # 保存conv输出的空间维度用于后续重塑
        _, C_conv, T_conv, H_conv, W_conv = hidden_states.shape
        
        # 将3D特征重塑为2D进行注意力计算
        hidden_states_2d = rearrange(hidden_states, 'b c t h w -> (b t) c h w')
        print(f"hidden_states_2d shape: {hidden_states_2d.shape}")
        
        # 重塑为适合注意力的格式
        spatial_features = rearrange(hidden_states_2d, 'bt c h w -> bt (h w) c')
        print(f"spatial_features shape: {spatial_features.shape}")
        
        # 创建时间特征 - 使用空间池化
        spatial_pooled = F.adaptive_avg_pool2d(hidden_states_2d, (1, 1)).squeeze(-1).squeeze(-1)  # [B*T, C]
        temporal_features = rearrange(spatial_pooled, '(b t) c -> b t c', b=B, t=T)
        print(f"temporal_features shape: {temporal_features.shape}")
        
        # 确保特征维度正确
        current_dim = spatial_features.shape[-1]  # 应该是C_conv
        if current_dim != self.config.cross_attention_dim:
            print(f"Adding projection layer: {current_dim} -> {self.config.cross_attention_dim}")
            if not hasattr(self, 'spatial_proj'):
                self.spatial_proj = nn.Linear(current_dim, self.config.cross_attention_dim).to(spatial_features.device)
            spatial_features = self.spatial_proj(spatial_features)
            
        if temporal_features.shape[-1] != self.config.cross_attention_dim:
            if not hasattr(self, 'temporal_proj'):
                self.temporal_proj = nn.Linear(temporal_features.shape[-1], self.config.cross_attention_dim).to(temporal_features.device)
            temporal_features = self.temporal_proj(temporal_features)
        
        print(f"Final spatial_features shape: {spatial_features.shape}")
        print(f"Final temporal_features shape: {temporal_features.shape}")
        
        # 6. 上下文增强音频学习
        spatial_out, temporal_out = self.context_audio_learning(
            spatial_features,
            temporal_features,
            audio_embeddings,
            face_mask
        )
        
        # 重塑回3D - 使用正确的维度
        if not hasattr(self, 'spatial_back_proj'):
            self.spatial_back_proj = nn.Linear(self.config.cross_attention_dim, current_dim).to(spatial_out.device)
        
        spatial_out_projected = self.spatial_back_proj(spatial_out)
        
        print(f"Reshaping spatial_out_projected: {spatial_out_projected.shape}")
        print(f"Target dimensions: bt={B*T}, c={current_dim}, h={H_conv}, w={W_conv}")
        print(f"Expected h*w: {H_conv * W_conv}, actual spatial dim: {spatial_out_projected.shape[1]}")
        
        # 重塑为3D特征
        spatial_out_3d = rearrange(spatial_out_projected, 'bt (h w) c -> bt c h w', h=H_conv, w=W_conv)
        hidden_states_reconstructed = rearrange(spatial_out_3d, '(b t) c h w -> b c t h w', b=B, t=T)
        
        # 7. 添加运动控制嵌入
        motion_embed = motion_embeddings['combined']  # [B, D]
        # 确保运动嵌入的维度与隐藏状态匹配
        if motion_embed.shape[-1] != current_dim:
            if not hasattr(self, 'motion_proj'):
                self.motion_proj = nn.Linear(motion_embed.shape[-1], current_dim).to(motion_embed.device)
            motion_embed = self.motion_proj(motion_embed)
        
        motion_embed = repeat(motion_embed, 'b d -> b d t h w', t=T, h=H_conv, w=W_conv)
        hidden_states = hidden_states_reconstructed + motion_embed
        
        # 8. 输出投影
        output = self.conv_out(hidden_states)
        
        return output
    
    def get_timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """获取时间步嵌入"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

# ==================== Sonic Pipeline ====================
class SonicPipeline:
    """Sonic推理管道"""
    def __init__(self, config: SonicConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型组件
        self.unet = SonicUNet(config).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="vae"
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="scheduler"
        )
        
        # CLIP处理器
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="feature_extractor"
        )
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="image_encoder"
        ).to(self.device)
        
    @torch.no_grad()
    def generate(self,
                 reference_image: np.ndarray,
                 audio_features: torch.Tensor,
                 num_frames: Optional[int] = None,
                 motion_buckets: Optional[Dict[str, int]] = None,
                 seed: int = 42) -> np.ndarray:
        """
        生成音频驱动的视频
        
        Args:
            reference_image: [H, W, 3] 参考图像
            audio_features: [T_audio, D] 音频特征
            num_frames: 生成帧数
            motion_buckets: 手动运动控制
            seed: 随机种子
        
        Returns:
            video: [T, H, W, 3] 生成的视频
        """
        if num_frames is None:
            num_frames = self.config.num_frames
            
        # 设置随机种子
        torch.manual_seed(seed)
        
        # 1. 预处理参考图像
        reference_image = cv2.resize(reference_image, (self.config.resolution, self.config.resolution))
        reference_tensor = torch.from_numpy(reference_image).permute(2, 0, 1).float() / 255.0
        reference_tensor = reference_tensor.unsqueeze(0).to(self.device)
        
        # 2. 提取CLIP特征
        clip_input = self.clip_processor(images=reference_image, return_tensors="pt")
        clip_features = self.clip_model(clip_input['pixel_values'].to(self.device)).image_embeds
        
        # 3. 编码参考图像
        reference_latent = self.vae.encode(reference_tensor * 2 - 1).latent_dist.sample()
        reference_latent = reference_latent * 0.18215
        
        # 4. 准备潜在表示
        latent_shape = (1, 4, num_frames, 
                       self.config.resolution // 8, 
                       self.config.resolution // 8)
        latents = torch.randn(latent_shape, device=self.device)
        
        # 5. 设置调度器
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        
        # 6. 扩散去噪循环
        for t in self.scheduler.timesteps:
            # 准备输入
            latent_model_input = torch.cat([latents, reference_latent.unsqueeze(2)], dim=1)
            
            # 无条件引导
            if self.config.guidance_scale_audio > 1.0:
                latent_model_input = torch.cat([latent_model_input] * 2)
                audio_features_input = torch.cat([
                    audio_features.unsqueeze(0),
                    torch.zeros_like(audio_features).unsqueeze(0)
                ])
            else:
                audio_features_input = audio_features.unsqueeze(0)
                
            # UNet预测
            noise_pred = self.unet(
                latent_model_input,
                t.unsqueeze(0),
                encoder_hidden_states={'reference': reference_latent},
                audio_features=audio_features_input,
                clip_features=clip_features,
                manual_motion_buckets=motion_buckets
            )
            
            # 条件引导
            if self.config.guidance_scale_audio > 1.0:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.guidance_scale_audio * (
                    noise_pred_cond - noise_pred_uncond
                )
                
            # 调度器步进
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # 7. 解码潜在表示
        latents = latents / 0.18215
        video_tensor = self.vae.decode(latents).sample
        
        # 8. 后处理
        video_tensor = (video_tensor + 1) / 2
        video_tensor = torch.clamp(video_tensor, 0, 1)
        video_tensor = video_tensor.squeeze(0).permute(1, 2, 3, 0)
        video_array = (video_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        return video_array

# ==================== 工具函数 ====================
def extract_audio_features(audio_path: str, model_name: str = "openai/whisper-tiny") -> torch.Tensor:
    """使用Whisper提取音频特征"""
    from transformers import WhisperProcessor, WhisperModel
    import librosa
    
    # 加载Whisper
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)
    
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 处理音频
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    
    # 提取特征
    with torch.no_grad():
        outputs = model(**inputs)
        # 获取多层特征并拼接
        features = []
        for i in range(len(outputs.encoder_hidden_states)):
            features.append(outputs.encoder_hidden_states[i])
        
        # 拼接最后5层
        audio_features = torch.cat(features[-5:], dim=-1)
        audio_features = audio_features.squeeze(0)  # [T, D]
        
    return audio_features

def detect_face_region(image: np.ndarray) -> Optional[np.ndarray]:
    """检测面部区域并返回掩码"""
    import face_recognition
    
    # 检测面部位置
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) == 0:
        return None
        
    # 创建掩码
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    for top, right, bottom, left in face_locations:
        # 扩展边界框
        margin = 20
        top = max(0, top - margin)
        bottom = min(h, bottom + margin)
        left = max(0, left - margin)
        right = min(w, right + margin)
        
        mask[top:bottom, left:right] = 1.0
        
    # 高斯模糊使边缘平滑
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    return mask

# ==================== 示例使用 ====================
def main():
    """示例：如何使用Sonic生成音频驱动的视频"""
    
    # 1. 初始化配置和管道
    config = SonicConfig()
    pipeline = SonicPipeline(config)
    
    # 2. 加载模型权重（假设已下载）
    # checkpoint = torch.load("checkpoints/Sonic/unet.pth")
    # pipeline.unet.load_state_dict(checkpoint)
    
    # 3. 准备输入
    # 加载参考图像
    reference_image = cv2.imread("path/to/reference_image.jpg")
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    
    # 提取音频特征
    audio_features = extract_audio_features("path/to/audio.wav")
    
    # 检测面部区域
    face_mask = detect_face_region(reference_image)
    
    # 4. 生成视频
    print("生成视频中...")
    video = pipeline.generate(
        reference_image=reference_image,
        audio_features=audio_features,
        num_frames=25,
        motion_buckets={
            'translation': 64,  # 中等头部运动
            'expression': 96    # 较强表情
        }
    )
    
    # 5. 保存视频
    print("保存视频...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 25.0, 
                         (config.resolution, config.resolution))
    
    for frame in video:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
    out.release()
    print("视频生成完成！")

if __name__ == "__main__":
    main()
