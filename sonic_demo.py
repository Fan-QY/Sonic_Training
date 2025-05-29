"""
Sonic推理演示脚本
使用训练好的Sonic模型生成音频驱动的肖像动画
"""

import os
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import WhisperProcessor, WhisperModel
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import gradio as gr
from typing import Optional, Tuple
import tempfile
import warnings
warnings.filterwarnings('ignore')

# 导入Sonic组件
from sonic_implementation import SonicUNet, SonicConfig, SonicPipeline

class SonicDemo:
    """Sonic演示类"""
    def __init__(self, checkpoint_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载模型
        self._load_models(checkpoint_path)
        
    def _load_models(self, checkpoint_path: str):
        """加载所有必需的模型"""
        print("Loading models...")
        
        # 初始化配置
        self.config = SonicConfig()
        
        # 加载UNet
        self.unet = SonicUNet(self.config).to(self.device)
        unet_path = os.path.join(checkpoint_path, "unet.pth")
        if os.path.exists(unet_path):
            state_dict = torch.load(unet_path, map_location=self.device)
            self.unet.load_state_dict(state_dict)
            print(f"Loaded UNet from {unet_path}")
        else:
            print("Warning: UNet checkpoint not found, using random weights")
        
        self.unet.eval()
        
        # 加载VAE
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="vae"
        ).to(self.device)
        self.vae.eval()
        
        # 加载CLIP
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="feature_extractor"
        )
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="image_encoder"
        ).to(self.device)
        self.clip_model.eval()
        
        # 加载Whisper
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny").to(self.device)
        self.whisper_model.eval()
        
        # 加载调度器
        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="scheduler"
        )
        
        print("All models loaded successfully!")
    
    @torch.no_grad()
    def extract_audio_features(self, audio_path: str) -> torch.Tensor:
        """提取音频特征"""
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 使用Whisper处理
        inputs = self.whisper_processor(
            audio, 
            return_tensors="pt", 
            sampling_rate=16000
        ).to(self.device)
        
        # 提取特征
        outputs = self.whisper_model(**inputs)
        
        # 获取多层特征
        features = []
        for hidden_state in outputs.encoder_hidden_states[-5:]:  # 最后5层
            features.append(hidden_state)
        
        # 拼接特征
        audio_features = torch.cat(features, dim=-1)
        audio_features = audio_features.squeeze(0)  # [T, D]
        
        return audio_features
    
    @torch.no_grad()
    def generate_video(self,
                      image_path: str,
                      audio_path: str,
                      output_path: str,
                      num_frames: int = 25,
                      fps: int = 25,
                      motion_intensity: str = "moderate",
                      seed: int = 42) -> str:
        """生成视频"""
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 加载和预处理图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config.resolution, self.config.resolution))
        
        # 图像归一化
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 提取CLIP特征
        clip_input = self.clip_processor(images=image, return_tensors="pt")
        clip_features = self.clip_model(clip_input['pixel_values'].to(self.device)).image_embeds
        
        # 编码参考图像
        reference_latent = self.vae.encode(image_tensor).latent_dist.sample()
        reference_latent = reference_latent * 0.18215
        
        # 提取音频特征
        print("Extracting audio features...")
        audio_features = self.extract_audio_features(audio_path)
        
        # 设置运动强度
        motion_scales = {
            "mild": 1.0,
            "moderate": 1.5,
            "intense": 2.0
        }
        self.config.dynamic_scale = motion_scales.get(motion_intensity, 1.5)
        
        # 准备潜在表示
        latent_shape = (1, 4, num_frames, 
                       self.config.resolution // 8, 
                       self.config.resolution // 8)
        latents = torch.randn(latent_shape, device=self.device)
        
        # 设置调度器
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        
        # 扩散去噪循环
        print("Generating video...")
        for i, t in enumerate(self.scheduler.timesteps):
            # 准备模型输入
            latent_model_input = torch.cat([latents, reference_latent.unsqueeze(2).expand(-1, -1, num_frames, -1, -1)], dim=1)
            
            # 无条件引导
            if self.config.guidance_scale_audio > 1.0:
                latent_model_input = torch.cat([latent_model_input] * 2)
                audio_features_input = torch.cat([
                    audio_features.unsqueeze(0),
                    torch.zeros_like(audio_features).unsqueeze(0)
                ])
                clip_features_input = torch.cat([clip_features, torch.zeros_like(clip_features)])
            else:
                audio_features_input = audio_features.unsqueeze(0)
                clip_features_input = clip_features
            
            # 预测噪声
            noise_pred = self.unet(
                latent_model_input,
                t.unsqueeze(0).to(self.device),
                encoder_hidden_states={'reference': reference_latent},
                audio_features=audio_features_input,
                clip_features=clip_features_input
            )
            
            # 应用引导
            if self.config.guidance_scale_audio > 1.0:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.guidance_scale_audio * (
                    noise_pred_cond - noise_pred_uncond
                )
            
            # 调度器步进
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # 显示进度
            if i % 5 == 0:
                print(f"Progress: {i+1}/{len(self.scheduler.timesteps)}")
        
        # 解码视频
        print("Decoding video...")
        latents = latents / 0.18215
        video_tensor = self.vae.decode(latents).sample
        
        # 后处理
        video_tensor = (video_tensor + 1) / 2
        video_tensor = torch.clamp(video_tensor, 0, 1)
        video_tensor = video_tensor.squeeze(0).permute(1, 2, 3, 0)  # [T, H, W, C]
        video_array = (video_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # 保存视频
        print("Saving video...")
        temp_video_path = output_path.replace('.mp4', '_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, 
                             (self.config.resolution, self.config.resolution))
        
        for frame in video_array:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        # 添加音频
        print("Adding audio...")
        video_clip = VideoFileClip(temp_video_path)
        audio_clip = AudioFileClip(audio_path)
        
        # 调整音频长度以匹配视频
        video_duration = video_clip.duration
        if audio_clip.duration > video_duration:
            audio_clip = audio_clip.subclip(0, video_duration)
        
        # 合成最终视频
        final_video = video_clip.set_audio(audio_clip)
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        # 清理临时文件
        os.remove(temp_video_path)
        video_clip.close()
        audio_clip.close()
        
        print(f"Video saved to {output_path}")
        return output_path

def create_gradio_interface(demo: SonicDemo):
    """创建Gradio界面"""
    def process_video(image, audio, motion_intensity, seed):
        """处理视频生成请求"""
        try:
            # 保存上传的文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_img:
                image.save(tmp_img.name)
                image_path = tmp_img.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                # 保存音频文件
                sf.write(tmp_audio.name, audio[1], audio[0])
                audio_path = tmp_audio.name
            
            # 生成输出路径
            output_path = tempfile.mktemp(suffix='.mp4')
            
            # 生成视频
            result_path = demo.generate_video(
                image_path=image_path,
                audio_path=audio_path,
                output_path=output_path,
                motion_intensity=motion_intensity,
                seed=seed
            )
            
            # 清理临时文件
            os.remove(image_path)
            os.remove(audio_path)
            
            return result_path
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    # 创建界面
    with gr.Blocks(title="Sonic - 音频驱动肖像动画") as interface:
        gr.Markdown("""
        # 🎵 Sonic - 音频驱动肖像动画
        
        上传一张肖像图片和一段音频，生成与音频同步的说话视频。
        
        **提示：**
        - 图片最好是正面肖像，背景简单
        - 音频最好是清晰的说话声
        - 生成过程可能需要1-2分钟
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="肖像图片", type="pil")
                audio_input = gr.Audio(label="音频文件", type="numpy")
                
                with gr.Row():
                    motion_intensity = gr.Radio(
                        choices=["mild", "moderate", "intense"],
                        value="moderate",
                        label="运动强度"
                    )
                    seed = gr.Slider(
                        minimum=0,
                        maximum=999999,
                        value=42,
                        step=1,
                        label="随机种子"
                    )
                
                generate_btn = gr.Button("生成视频", variant="primary")
                
            with gr.Column():
                video_output = gr.Video(label="生成结果")
        
        # 示例
        gr.Examples(
            examples=[
                ["examples/portrait1.jpg", "examples/audio1.wav", "moderate", 42],
                ["examples/portrait2.jpg", "examples/audio2.wav", "intense", 123],
            ],
            inputs=[image_input, audio_input, motion_intensity, seed],
            outputs=video_output,
            fn=process_video,
            cache_examples=True
        )
        
        # 绑定事件
        generate_btn.click(
            fn=process_video,
            inputs=[image_input, audio_input, motion_intensity, seed],
            outputs=video_output
        )
    
    return interface

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Sonic Demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--audio", type=str, help="Path to input audio")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--motion", type=str, default="moderate", 
                       choices=["mild", "moderate", "intense"], help="Motion intensity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio interface")
    parser.add_argument("--share", action="store_true", help="Share Gradio interface")
    
    args = parser.parse_args()
    
    # 初始化演示
    demo = SonicDemo(args.checkpoint)
    
    if args.gradio:
        # 启动Gradio界面
        interface = create_gradio_interface(demo)
        interface.launch(share=args.share)
    else:
        # 命令行模式
        if not args.image or not args.audio:
            print("Error: Please provide both --image and --audio paths")
            return
        
        demo.generate_video(
            image_path=args.image,
            audio_path=args.audio,
            output_path=args.output,
            motion_intensity=args.motion,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
