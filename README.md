# Sonic_Training

# 📖 概述
Sonic是一个基于音频驱动的视频生成模型，可以将静态图像转换为与音频同步的动态视频。原作者只开源了推理代码，本项目主要为了复现其训练代码 “Sonic: Shifting Focus to Global Audio Perception in Portrait Animation, CVPR 2025” 原作者所属单位为腾讯和浙江大学


## Sonic: Shifting Focus to Global Audio Perception in Portrait Animation

基于CVPR 2025论文的音频驱动肖像动画实现。

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (推荐使用GPU)
- 至少32GB GPU内存（推荐）

### 安装依赖

```bash
# 创建虚拟环境
conda create -n sonic python=3.8
conda activate sonic

# 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install diffusers transformers accelerate
pip install opencv-python moviepy librosa soundfile
pip install gradio wandb tqdm einops
pip install face_recognition
```

### 下载预训练模型

1. **下载Sonic模型权重**：
```bash
# 使用huggingface-cli
pip install huggingface_hub[cli]
huggingface-cli download LeonJoe13/Sonic --local-dir checkpoints
```

2. **下载依赖模型**：
```bash
# SVD模型
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir checkpoints/stable-video-diffusion-img2vid-xt

# Whisper模型
huggingface-cli download openai/whisper-tiny --local-dir checkpoints/whisper-tiny
```

## 💻 使用方法

### 1. 推理演示

#### 命令行方式
```bash
python sonic_demo.py \
    --checkpoint checkpoints/Sonic \
    --image examples/portrait.jpg \
    --audio examples/audio.wav \
    --output output.mp4 \
    --motion moderate \
    --seed 42
```

#### Gradio界面
```bash
python sonic_demo.py --checkpoint checkpoints/Sonic --gradio --share
```

### 2. 训练模型

准备训练数据（MP4视频文件），然后运行：

```bash
python sonic_training.py \
    --model_path checkpoints/stable-video-diffusion-img2vid-xt \
    --data_root /path/to/training/videos \
    --output_dir /path/to/save/checkpoints \
    --train_batch_size 1 \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
    --mixed_precision fp16 \
    --num_frames 25 \
    --resolution 576 \
    --use_wandb
```

### 3. 在代码中使用

```python
from sonic_implementation import SonicConfig, SonicPipeline
import cv2

# 初始化
config = SonicConfig()
pipeline = SonicPipeline(config)

# 加载图像和音频
reference_image = cv2.imread("portrait.jpg")
audio_features = extract_audio_features("audio.wav")

# 生成视频
video = pipeline.generate(
    reference_image=reference_image,
    audio_features=audio_features,
    num_frames=25,
    motion_buckets={'translation': 64, 'expression': 96}
)

# 保存视频
save_video(video, "output.mp4")
```

## 📊 训练细节

### 数据准备

训练数据应该是包含清晰音频和面部画面的MP4视频文件：
- 建议视频长度：5-30秒
- 分辨率：至少512x512
- 音频：清晰的说话声
- 本项目中我们主要采用了CelebV-HQ的数据进行了训练。对于相应的数据集下载和处理请参照CelebV-HQ (https://github.com/CelebV-HQ/CelebV-HQ) 的原始方案。


目录结构：
```
data_root/
├── video1.mp4
├── video2.mp4
├── ...
└── videoN.mp4
```

### 训练参数说明

- `train_batch_size`: 批次大小（建议1，因为视频占用内存大）
- `gradient_accumulation_steps`: 梯度累积步数（用于模拟更大批次）
- `num_frames`: 每个训练样本的帧数（默认25）
- `resolution`: 训练分辨率（默认576）
- `mixed_precision`: 混合精度训练（fp16推荐）

### 运动控制参数

- **Translation Bucket** (0-127): 控制头部运动幅度
  - 0-32: 轻微运动
  - 32-96: 中等运动
  - 96-127: 剧烈运动

- **Expression Bucket** (0-127): 控制表情变化强度
  - 0-32: 细微表情
  - 32-96: 正常表情
  - 96-127: 夸张表情

- **Dynamic Scale**: 整体运动强度
  - 1.0: 温和（mild）
  - 1.5: 适中（moderate）
  - 2.0: 强烈（intense）

## 🔧 高级功能

### 自定义音频特征提取

```python
def custom_audio_features(audio_path):
    # 使用自定义的音频处理
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 提取MFCC、音高等特征
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    pitch = librosa.piptrack(y=audio, sr=sr)
    
    # 组合特征
    features = combine_features(mfcc, pitch)
    return features
```

### 批量处理

```python
def batch_process(image_paths, audio_paths, output_dir):
    for img_path, audio_path in zip(image_paths, audio_paths):
        output_path = f"{output_dir}/{Path(img_path).stem}.mp4"
        pipeline.generate_video(
            image_path=img_path,
            audio_path=audio_path,
            output_path=output_path
        )
```

## 📝 引用

如果您使用了本代码，请引用原始论文：

```bibtex
@article{ji2024sonic,
    title={Sonic: Shifting Focus to Global Audio Perception in Portrait Animation},
    author={Ji, Xiaozhong and Hu, Xiaobin and Xu, Zhihong and Zhu, Junwei and Lin, Chuming and He, Qingdong and Zhang, Jiangning and Luo, Donghao and Chen, Yi and Lin, Qin and others},
    journal={arXiv preprint arXiv:2411.16331},
    year={2024}
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用非商业许可证。如需商业使用，请联系原作者。

## 🙏 致谢

- 感谢Stable Video Diffusion团队提供的基础模型
- 感谢OpenAI提供的Whisper模型
- 感谢所有开源社区的贡献者

## 常见问题

### Q: 内存不足怎么办？
A: 尝试以下方法：
- 减小`resolution`到512或更小
- 减少`num_frames`
- 使用`gradient_accumulation_steps`
- 启用混合精度训练`--mixed_precision fp16`

### Q: 生成的视频不够流畅？
A: 可以尝试：
- 增加`shift_offset`参数（默认3）
- 调整运动强度参数
- 使用更高质量的训练数据

### Q: 如何提高唇同步质量？
A: 
- 确保音频质量清晰
- 调整`guidance_scale_audio`参数
- 使用更多包含清晰唇动的训练数据
