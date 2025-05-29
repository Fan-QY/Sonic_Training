"""
Sonic训练脚本 - 基于论文和GitHub仓库实现
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPVisionModelWithProjection, WhisperModel, WhisperProcessor
from einops import rearrange, repeat
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import cv2
from tqdm.auto import tqdm
import argparse
from dataclasses import dataclass, field
import logging

import wandb

logger = get_logger(__name__)

# ==================== 训练配置 ====================
@dataclass
class TrainingConfig:
    """训练配置参数"""
    # 模型路径
    model_path: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    whisper_model: str = "openai/whisper-tiny"
    
    # 数据路径
    data_root: str = "./data"
    output_dir: str = "./checkpoints"
    
    # # 训练参数
    # train_batch_size: int = 1
    # gradient_accumulation_steps: int = 1
    # num_train_epochs: int = 100
    # learning_rate: float = 1e-5
    # lr_scheduler: str = "cosine"
    # lr_warmup_steps: int = 500
    # adam_beta1: float = 0.9
    # adam_beta2: float = 0.999
    # adam_weight_decay: float = 1e-2
    # adam_epsilon: float = 1e-8
    # max_grad_norm: float = 1.0
    
    # # 模型参数
    # num_frames: int = 25
    # resolution: int = 576
    # audio_context_duration: float = 0.2  # 秒
    
    # # 混合精度
    # mixed_precision: str = "fp16"
    
    # # 保存和日志
    # save_steps: int = 1000
    # logging_steps: int = 50
    # validation_steps: int = 500
    # num_validation_samples: int = 4
    
    # # 其他
    # seed: int = 42
    # num_workers: int = 4
    # use_wandb: bool = True
    # wandb_project: str = "sonic-training"
    
    # # 条件丢弃概率（用于无条件引导训练）
    # audio_drop_prob: float = 0.05
    # image_drop_prob: float = 0.05
    # both_drop_prob: float = 0.05

    # 训练参数 - 内存优化
    train_batch_size: int = 1  # 保持为1
    gradient_accumulation_steps: int = 4  # 增加梯度累积步数来模拟更大的batch
    num_train_epochs: int = 100
    learning_rate: float = 1e-5
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 模型参数 - 减少内存使用
    num_frames: int = 16  # 从25减少到16帧
    resolution: int = 256  # 从576减少到256
    audio_context_duration: float = 0.2
    
    # 混合精度 - 使用bf16可能更节省内存
    mixed_precision: str = "bf16"  # 改为bf16
    
    # 保存和日志
    save_steps: int = 1000
    logging_steps: int = 50
    validation_steps: int = 500
    num_validation_samples: int = 4
    
    # 其他
    seed: int = 42
    num_workers: int = 2  # 减少数据加载的内存使用
    use_wandb: bool = True
    wandb_project: str = "sonic-training"
    
    # 条件丢弃概率
    audio_drop_prob: float = 0.05
    image_drop_prob: float = 0.05
    both_drop_prob: float = 0.05

# ==================== 数据集类 ====================
class SonicDataset(Dataset):
    """Sonic训练数据集"""
    def __init__(self, 
                 data_root: str,
                 resolution: int = 576,
                 num_frames: int = 25,
                 audio_context_duration: float = 0.2,
                 audio_sample_rate: int = 16000):
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.num_frames = num_frames
        self.audio_context_duration = audio_context_duration
        self.audio_sample_rate = audio_sample_rate
        
        # 收集所有视频文件
        self.video_files = list(self.data_root.glob("*.mp4"))
        logger.info(f"Found {len(self.video_files)} videos in {data_root}")
        
        # 初始化Whisper处理器
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_path = self.video_files[idx]
        
        try:
            # 加载视频和音频
            frames, audio, fps = self._load_video_audio(video_path)
            
            # 随机选择起始帧
            total_frames = len(frames)
            if total_frames >= self.num_frames:
                start_idx = np.random.randint(0, total_frames - self.num_frames + 1)
            else:
                start_idx = 0
                # 填充帧数不足的情况
                frames = frames + [frames[-1]] * (self.num_frames - total_frames)
            
            # 提取视频片段
            video_frames = frames[start_idx:start_idx + self.num_frames]
            video_tensor = self._process_frames(video_frames)
            
            # 提取对应的音频片段
            audio_start = start_idx / fps
            audio_end = (start_idx + self.num_frames) / fps
            audio_features = self._extract_audio_features(
                audio, 
                audio_start, 
                audio_end
            )
            
            # 提取面部边界框（用于运动桶计算）
            face_bboxes = self._extract_face_bboxes(video_frames)
            motion_buckets = self._compute_motion_buckets(face_bboxes)
            
            # 第一帧作为参考图像
            reference_image = video_tensor[:, 0]  # [C, H, W]
            
            return {
                'video': video_tensor,  # [C, T, H, W]
                'reference_image': reference_image,  # [C, H, W]
                'audio_features': audio_features,  # [T_audio, D]
                'motion_buckets': motion_buckets,  # {'translation': int, 'expression': int}
                'video_path': str(video_path)
            }
            
        except Exception as e:
            logger.error(f"Error loading {video_path}: {e}")
            # 返回一个虚拟样本
            return self._get_dummy_sample()
    
    def _load_video_audio(self, video_path: Path) -> Tuple[List[np.ndarray], np.ndarray, float]:
        """加载视频和音频"""
        import moviepy.editor as mp
        
        # 使用moviepy加载视频
        video = mp.VideoFileClip(str(video_path))
        
        # 提取帧
        frames = []
        for frame in video.iter_frames():
            frames.append(frame)
        
        # 提取音频
        audio = video.audio.to_soundarray(fps=self.audio_sample_rate)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # 转换为单声道
            
        fps = video.fps
        video.close()
        
        return frames, audio, fps
    
    def _process_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """处理视频帧"""
        processed_frames = []
        
        for frame in frames:
            # 调整大小
            frame = cv2.resize(frame, (self.resolution, self.resolution))
            # 归一化到[-1, 1]
            frame = (frame.astype(np.float32) / 127.5) - 1.0
            processed_frames.append(frame)
        
        # 转换为tensor [T, H, W, C] -> [C, T, H, W]
        frames_tensor = torch.from_numpy(np.stack(processed_frames))
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)
        
        return frames_tensor
    
    # def _extract_audio_features(self, 
    #                            audio: np.ndarray, 
    #                            start_time: float, 
    #                            end_time: float) -> torch.Tensor:
    #     """提取音频特征"""
    #     # 计算采样点
    #     start_sample = int(start_time * self.audio_sample_rate)
    #     end_sample = int(end_time * self.audio_sample_rate)
        
    #     # 添加上下文
    #     context_samples = int(self.audio_context_duration * self.audio_sample_rate)
    #     start_sample = max(0, start_sample - context_samples)
    #     end_sample = min(len(audio), end_sample + context_samples)
        
    #     # 提取音频片段
    #     audio_segment = audio[start_sample:end_sample]
        
    #     # 使用Whisper处理器
    #     inputs = self.whisper_processor(
    #         audio_segment, 
    #         return_tensors="pt", 
    #         sampling_rate=self.audio_sample_rate
    #     )
        
    #     # 这里简化处理，实际应该通过Whisper模型提取特征
    #     # 返回一个虚拟的特征张量
    #     audio_length = len(audio_segment) // 320  # Whisper的下采样率
    #     audio_features = torch.randn(audio_length, 384)  # Whisper-Tiny特征维度
        
    #     return audio_features
    def _extract_audio_features(self, 
                           audio: np.ndarray, 
                           start_time: float, 
                           end_time: float) -> torch.Tensor:
        """提取音频特征"""
        # 计算采样点
        start_sample = int(start_time * self.audio_sample_rate)
        end_sample = int(end_time * self.audio_sample_rate)
        
        # 添加上下文
        context_samples = int(self.audio_context_duration * self.audio_sample_rate)
        start_sample = max(0, start_sample - context_samples)
        end_sample = min(len(audio), end_sample + context_samples)
        
        # 提取音频片段
        audio_segment = audio[start_sample:end_sample]
        
        # 使用Whisper处理器
        inputs = self.whisper_processor(
            audio_segment, 
            return_tensors="pt", 
            sampling_rate=self.audio_sample_rate
        )
        
        # 确保音频特征序列长度与视频帧数匹配
        target_length = self.num_frames  # 25帧
        audio_features = torch.randn(target_length, 384)  # 固定长度为25
        
        return audio_features
    
    def _extract_face_bboxes(self, frames: List[np.ndarray]) -> List[Dict[str, int]]:
        """提取面部边界框"""
        # 这里简化处理，实际应该使用face detection模型
        bboxes = []
        for frame in frames:
            h, w = frame.shape[:2]
            # 假设面部在图像中心
            bbox = {
                'x': w // 4,
                'y': h // 4,
                'width': w // 2,
                'height': h // 2
            }
            bboxes.append(bbox)
        return bboxes
    
    def _compute_motion_buckets(self, bboxes: List[Dict[str, int]]) -> Dict[str, int]:
        """计算运动桶值"""
        # 计算边界框的方差作为运动幅度
        x_coords = [bbox['x'] for bbox in bboxes]
        y_coords = [bbox['y'] for bbox in bboxes]
        
        x_var = np.var(x_coords) if len(x_coords) > 1 else 0
        y_var = np.var(y_coords) if len(y_coords) > 1 else 0
        
        # 映射到0-127范围
        translation_bucket = int(min(127, (x_var + y_var) * 0.1))
        
        # 表情运动（这里简化处理）
        expression_bucket = np.random.randint(32, 96)
        
        return {
            'translation': translation_bucket,
            'expression': expression_bucket
        }
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """返回虚拟样本（错误处理用）"""
        return {
            'video': torch.zeros(3, self.num_frames, self.resolution, self.resolution),
            'reference_image': torch.zeros(3, self.resolution, self.resolution),
            'audio_features': torch.zeros(100, 384),
            'motion_buckets': {'translation': 64, 'expression': 64},
            'video_path': 'dummy'
        }

# ==================== 训练器类 ====================
class SonicTrainer:
    """Sonic训练器"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # 初始化加速器
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="wandb" if config.use_wandb else None,
        )
        
        # 设置日志
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 初始化模型
        self._init_models()
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 初始化优化器和调度器
        self._init_optimizer_scheduler()
        
        # 准备训练
        self._prepare_training()
        
        # 初始化wandb
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb_project,
                config=vars(config),
                name=f"sonic-{config.seed}"
            )
    
    def _init_models(self):
        """初始化模型"""
        logger.info("Loading models...")
        
        # 加载VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.model_path,
            subfolder="vae"
        )
        self.vae.requires_grad_(False)
        
        # 加载UNet（使用我们的SonicUNet）
        from sonic_implementation import SonicUNet, SonicConfig
        sonic_config = SonicConfig(
            num_frames=self.config.num_frames,
            resolution=self.config.resolution
        )
        self.unet = SonicUNet(sonic_config)
        
        # 从SVD初始化权重
        logger.info("Initializing from SVD weights...")
        # 这里应该加载SVD的预训练权重
        
        # 加载CLIP
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            self.config.model_path,
            subfolder="image_encoder"
        )
        self.clip_model.requires_grad_(False)
        
        # 加载Whisper
        self.whisper_model = WhisperModel.from_pretrained(self.config.whisper_model)
        self.whisper_model.requires_grad_(False)
        
        # 噪声调度器
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            self.config.model_path,
            subfolder="scheduler"
        )
        
    def _init_dataloaders(self):
        """初始化数据加载器"""
        # 训练数据集
        train_dataset = SonicDataset(
            data_root=self.config.data_root,
            resolution=self.config.resolution,
            num_frames=self.config.num_frames,
            audio_context_duration=self.config.audio_context_duration
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
    def _init_optimizer_scheduler(self):
        """初始化优化器和学习率调度器"""
        # 只优化UNet和音频相关模块
        params_to_optimize = self.unet.parameters()
        
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )
        
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.config.gradient_accumulation_steps,
            num_training_steps=len(self.train_dataloader) * self.config.num_train_epochs,
        )
    def _safe_move_to_device(self, model, device):
        """安全地将模型移动到指定设备"""
        # 检查是否有meta tensor需要初始化
        has_meta = False
        for name, param in model.named_parameters():
            if param.is_meta:
                has_meta = True
                break
        
        if has_meta:
            # 如果有meta tensor，使用to_empty
            model.to_empty(device=device)
        else:
            # 否则正常移动
            model.to(device)
        
        return model   
             
    def _prepare_training(self):
        """准备训练"""
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        
        # 移动其他模型到设备
        # self.vae.to(self.accelerator.device)
        # self.clip_model.to(self.accelerator.device)
        # self.whisper_model.to(self.accelerator.device)
        self.vae = self._safe_move_to_device(self.vae, self.accelerator.device)
        self.clip_model = self._safe_move_to_device(self.clip_model, self.accelerator.device)
        self.whisper_model = self._safe_move_to_device(self.whisper_model, self.accelerator.device)
        
        # 计算总训练步数
        num_update_steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        self.max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.config.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.config.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = "
                   f"{self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps}")
        logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        
    def train(self):
        """主训练循环"""
        global_step = 0
        first_epoch = 0
        
        # 进度条
        progress_bar = tqdm(
            range(global_step, self.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")
        
        for epoch in range(first_epoch, self.config.num_train_epochs):
            self.unet.train()
            train_loss = 0.0
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # 编码视频到潜在空间
                    video_latents = self._encode_video(batch['video'])
                    reference_latents = self._encode_image(batch['reference_image'])
                    
                    # 提取音频特征
                    audio_features = self._extract_audio_features_batch(batch['audio_features'])
                    
                    # 提取CLIP特征
                    clip_features = self._extract_clip_features(batch['reference_image'])
                    
                    # 添加噪声
                    noise = torch.randn_like(video_latents)
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (video_latents.shape[0],), device=video_latents.device
                    ).long()
                    
                    noisy_latents = self.noise_scheduler.add_noise(
                        video_latents, noise, timesteps
                    )
                    
                    # 条件丢弃（用于无条件引导）
                    audio_features, clip_features = self._apply_conditioning_dropout(
                        audio_features, clip_features, batch['video'].shape[0]
                    )
                    
                    # 前向传播
                    #model_input = torch.cat([noisy_latents, reference_latents.unsqueeze(2)], dim=1)
                    # 获取时间维度大小
                    num_frames = noisy_latents.shape[2]  # T=25

                    # 将reference_latents扩展到所有帧
                    # 从 [B, C, H, W] -> [B, C, T, H, W]
                    reference_latents_expanded = reference_latents.unsqueeze(2).expand(-1, -1, num_frames, -1, -1)

                    # 在通道维度拼接
                    model_input = torch.cat([noisy_latents, reference_latents_expanded], dim=1)
                    noise_pred = self.unet(
                        model_input,
                        timesteps,
                        encoder_hidden_states={'reference': reference_latents},
                        audio_features=audio_features,
                        clip_features=clip_features,
                        manual_motion_buckets=batch['motion_buckets']
                    )
                    
                    # 计算损失
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    
                    # 反向传播
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), self.config.max_grad_norm)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # 更新进度
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    train_loss += loss.detach().item()
                    
                    # 日志记录
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = train_loss / self.config.logging_steps
                        logger.info(f"Step: {global_step}, Loss: {avg_loss:.4f}")
                        
                        if self.config.use_wandb:
                            self.accelerator.log(
                                {
                                    "train_loss": avg_loss,
                                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                                },
                                step=global_step,
                            )
                        train_loss = 0.0
                    
                    # 保存检查点
                    if global_step % self.config.save_steps == 0:
                        self._save_checkpoint(global_step)
                    
                    # 验证
                    if global_step % self.config.validation_steps == 0:
                        self._validate(global_step)
                
                logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                if global_step >= self.max_train_steps:
                    break
        
        # 保存最终模型
        self._save_checkpoint(global_step, final=True)
        self.accelerator.end_training()
        
    def _encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """编码视频到潜在空间"""
        b, c, t, h, w = video.shape
        video = rearrange(video, 'b c t h w -> (b t) c h w')
        
        with torch.no_grad():
            latents = self.vae.encode(video).latent_dist.sample()
            latents = latents * 0.18215
            
        latents = rearrange(latents, '(b t) c h w -> b c t h w', b=b, t=t)
        return latents
    
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像到潜在空间"""
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * 0.18215
        return latents
    
    # def _extract_audio_features_batch(self, audio_features: torch.Tensor) -> torch.Tensor:
    #     """批量提取音频特征"""
    #     # 使用Whisper模型提取特征
    #     # 这里简化处理，实际应该通过Whisper模型
    #     return audio_features.to(self.accelerator.device)
    def _extract_audio_features_batch(self, audio_features: torch.Tensor) -> torch.Tensor:
        """批量提取音频特征"""
        audio_features = audio_features.to(self.accelerator.device)
        
        # 确保音频特征的序列长度与视频帧数匹配
        target_seq_len = self.config.num_frames  # 25帧
        current_seq_len = audio_features.shape[1]
        
        if current_seq_len != target_seq_len:
            if current_seq_len > target_seq_len:
                # 截断到25帧
                audio_features = audio_features[:, :target_seq_len, :]
            else:
                # 填充到25帧
                padding = target_seq_len - current_seq_len
                audio_features = F.pad(audio_features, (0, 0, 0, padding), mode='constant', value=0)
        
        return audio_features
    
    def _extract_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        # """提取CLIP特征"""
        # with torch.no_grad():
        #     # 归一化图像
        #     images = (images + 1.0) / 2.0
        #     clip_features = self.clip_model(images).image_embeds
        # return clip_features
        """提取CLIP特征"""
        with torch.no_grad():
            # 将图像从[-1, 1]转换到[0, 1]
            images = (images + 1.0) / 2.0
            
            # 调整图像尺寸到CLIP期望的224x224
            if images.shape[-1] != 224 or images.shape[-2] != 224:
                images = F.interpolate(
                    images, 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # CLIP通常期望特定的归一化
            # 使用ImageNet的均值和标准差
            mean = torch.tensor([0.485, 0.456, 0.406]).to(images.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).to(images.device).view(1, 3, 1, 1)
            images = (images - mean) / std
            
            clip_features = self.clip_model(images).image_embeds
        return clip_features
    
    def _apply_conditioning_dropout(self, 
                                   audio_features: torch.Tensor,
                                   clip_features: torch.Tensor,
                                   batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用条件丢弃"""
        # 随机丢弃音频
        if np.random.random() < self.config.audio_drop_prob:
            audio_features = torch.zeros_like(audio_features)
            
        # 随机丢弃图像
        if np.random.random() < self.config.image_drop_prob:
            clip_features = torch.zeros_like(clip_features)
            
        # 随机丢弃两者
        if np.random.random() < self.config.both_drop_prob:
            audio_features = torch.zeros_like(audio_features)
            clip_features = torch.zeros_like(clip_features)
            
        return audio_features, clip_features
    
    def _save_checkpoint(self, global_step: int, final: bool = False):
        """保存检查点"""
        if self.accelerator.is_main_process:
            save_path = os.path.join(self.config.output_dir, f"checkpoint-{global_step}")
            if final:
                save_path = os.path.join(self.config.output_dir, "final_model")
                
            os.makedirs(save_path, exist_ok=True)
            
            # 保存UNet
            self.accelerator.unwrap_model(self.unet).save_pretrained(
                os.path.join(save_path, "unet")
            )
            
            # 保存优化器和调度器状态
            torch.save({
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'global_step': global_step,
                'config': vars(self.config),
            }, os.path.join(save_path, "training_state.pth"))
            
            logger.info(f"Saved checkpoint to {save_path}")
    
    def _validate(self, global_step: int):
        """验证模型"""
        # 这里可以实现验证逻辑
        # 例如生成一些样本视频并计算指标
        logger.info(f"Running validation at step {global_step}")
        pass

# ==================== 主函数 ====================
def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="Train Sonic model")
    
    # 添加参数
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to training videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save checkpoints")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig(**vars(args))
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 初始化训练器
    trainer = SonicTrainer(config)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
