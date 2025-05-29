"""
Sonic训练数据下载和预处理脚本
包含数据集下载、视频处理、质量筛选等功能
"""

import os
import cv2
import numpy as np
import subprocess
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import requests
import zipfile
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import face_recognition
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
import logging
import shutil
from typing import List, Dict, Tuple, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 数据集下载 ====================
class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, root_dir: str = "./datasets"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exist_ok=True)
        
    def download_voxceleb2(self, username: str, password: str):
        """
        下载VoxCeleb2数据集
        需要在 https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ 注册账号
        """
        logger.info("Downloading VoxCeleb2...")
        voxceleb_dir = self.root_dir / "voxceleb2"
        voxceleb_dir.mkdir(exist_ok=True)
        
        # VoxCeleb2下载链接（需要认证）
        urls = [
            "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_partaa",
            "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4_partab",
            # ... 更多分片
        ]
        
        # 下载说明
        logger.info("""
        VoxCeleb2需要手动下载：
        1. 访问 https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html
        2. 注册账号并登录
        3. 下载所有 vox2_dev_mp4_part* 文件到 datasets/voxceleb2/
        4. 运行本脚本的解压功能
        """)
        
        return voxceleb_dir
    
    def download_celebv_hq(self):
        """下载CelebV-HQ数据集"""
        logger.info("Downloading CelebV-HQ...")
        celebv_dir = self.root_dir / "celebv_hq"
        celebv_dir.mkdir(exist_ok=True)
        
        # CelebV-HQ需要通过Google Drive下载
        logger.info("""
        CelebV-HQ下载步骤：
        1. 访问 https://github.com/CelebV-HQ/CelebV-HQ
        2. 按照README说明申请数据集访问权限
        3. 从Google Drive下载视频文件到 datasets/celebv_hq/
        """)
        
        return celebv_dir
    
    def download_vfhq(self):
        """下载VFHQ数据集"""
        logger.info("Downloading VFHQ...")
        vfhq_dir = self.root_dir / "vfhq"
        vfhq_dir.mkdir(exist_ok=True)
        
        logger.info("""
        VFHQ下载步骤：
        1. 访问 https://liangbinxie.github.io/projects/vfhq/
        2. 填写申请表获取下载链接
        3. 下载数据到 datasets/vfhq/
        """)
        
        return vfhq_dir
    
    def extract_archives(self, dataset_name: str):
        """解压数据集压缩包"""
        dataset_dir = self.root_dir / dataset_name
        
        # 查找所有压缩文件
        archives = list(dataset_dir.glob("*.zip")) + \
                  list(dataset_dir.glob("*.tar")) + \
                  list(dataset_dir.glob("*.tar.gz"))
        
        for archive in tqdm(archives, desc=f"Extracting {dataset_name}"):
            logger.info(f"Extracting {archive.name}...")
            
            if archive.suffix == '.zip':
                with zipfile.ZipFile(archive, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
            elif archive.suffix in ['.tar', '.gz']:
                with tarfile.open(archive, 'r:*') as tar_ref:
                    tar_ref.extractall(dataset_dir)
            
            # 删除压缩包以节省空间（可选）
            # archive.unlink()

# ==================== 视频预处理 ====================
class VideoPreprocessor:
    """视频预处理器"""
    
    def __init__(self, 
                 target_fps: int = 25,
                 target_resolution: int = 512,
                 min_duration: float = 2.0,
                 max_duration: float = 30.0):
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.min_duration = min_duration
        self.max_duration = max_duration
        
    def process_video(self, 
                     input_path: str, 
                     output_path: str,
                     crop_face: bool = True) -> bool:
        """
        处理单个视频
        - 调整分辨率和帧率
        - 裁剪面部区域
        - 检查音频质量
        """
        try:
            # 加载视频
            video = VideoFileClip(input_path)
            
            # 检查时长
            if video.duration < self.min_duration or video.duration > self.max_duration:
                logger.warning(f"Video duration {video.duration}s outside range [{self.min_duration}, {self.max_duration}]")
                video.close()
                return False
            
            # 检查是否有音频
            if video.audio is None:
                logger.warning(f"No audio in {input_path}")
                video.close()
                return False
            
            # 调整帧率
            if video.fps != self.target_fps:
                video = video.set_fps(self.target_fps)
            
            # 提取音频进行质量检查
            audio_array = video.audio.to_soundarray(fps=16000)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # 检查音频能量（过滤静音视频）
            audio_energy = np.sqrt(np.mean(audio_array**2))
            if audio_energy < 0.01:
                logger.warning(f"Audio too quiet in {input_path}")
                video.close()
                return False
            
            # 如果需要裁剪面部
            if crop_face:
                # 提取第一帧检测面部
                first_frame = video.get_frame(0)
                face_bbox = self._detect_face(first_frame)
                
                if face_bbox is None:
                    logger.warning(f"No face detected in {input_path}")
                    video.close()
                    return False
                
                # 裁剪视频
                video = self._crop_video_to_face(video, face_bbox)
            
            # 调整分辨率
            video = video.resize((self.target_resolution, self.target_resolution))
            
            # 保存处理后的视频
            video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                logger=None
            )
            
            video.close()
            return True
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False
    
    def _detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """检测面部并返回边界框"""
        # 使用face_recognition检测面部
        face_locations = face_recognition.face_locations(frame)
        
        if len(face_locations) == 0:
            return None
        
        # 选择最大的面部
        largest_face = max(face_locations, key=lambda x: (x[2]-x[0]) * (x[1]-x[3]))
        top, right, bottom, left = largest_face
        
        # 扩展边界框
        height, width = frame.shape[:2]
        margin = 0.3  # 30%边距
        
        face_height = bottom - top
        face_width = right - left
        
        top = max(0, int(top - face_height * margin))
        bottom = min(height, int(bottom + face_height * margin))
        left = max(0, int(left - face_width * margin))
        right = min(width, int(right + face_width * margin))
        
        # 确保是正方形
        face_size = max(bottom - top, right - left)
        center_y = (top + bottom) // 2
        center_x = (left + right) // 2
        
        top = max(0, center_y - face_size // 2)
        bottom = min(height, center_y + face_size // 2)
        left = max(0, center_x - face_size // 2)
        right = min(width, center_x + face_size // 2)
        
        return (top, left, bottom, right)
    
    def _crop_video_to_face(self, video: VideoFileClip, 
                           face_bbox: Tuple[int, int, int, int]) -> VideoFileClip:
        """根据面部边界框裁剪视频"""
        top, left, bottom, right = face_bbox
        return video.crop(x1=left, y1=top, x2=right, y2=bottom)
    
    def process_dataset(self, 
                       input_dir: str, 
                       output_dir: str,
                       num_workers: int = 4) -> Dict[str, int]:
        """批量处理数据集"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 收集所有视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        # 统计信息
        stats = {
            'total': len(video_files),
            'success': 0,
            'failed': 0,
            'no_face': 0,
            'no_audio': 0,
            'duration_invalid': 0
        }
        
        # 多线程处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for video_file in video_files:
                # 保持相对路径结构
                relative_path = video_file.relative_to(input_path)
                output_file = output_path / relative_path
                output_file.parent.mkdir(exist_ok=True, parents=True)
                
                future = executor.submit(
                    self.process_video,
                    str(video_file),
                    str(output_file),
                    crop_face=True
                )
                futures.append((future, video_file))
            
            # 处理结果
            for future, video_file in tqdm(futures, desc="Processing videos"):
                try:
                    success = future.result()
                    if success:
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1
                except Exception as e:
                    logger.error(f"Failed to process {video_file}: {e}")
                    stats['failed'] += 1
        
        # 保存处理统计
        stats_file = output_path / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing complete. Stats: {stats}")
        return stats

# ==================== 数据质量检查 ====================
class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self):
        self.min_sharpness = 20.0  # 最小清晰度阈值
        self.min_audio_snr = 10.0  # 最小信噪比(dB)
        
    def check_video_quality(self, video_path: str) -> Dict[str, float]:
        """检查视频质量"""
        cap = cv2.VideoCapture(video_path)
        
        metrics = {
            'sharpness': [],
            'brightness': [],
            'face_confidence': []
        }
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, frame_count // 10)  # 采样10帧
        
        for i in range(0, frame_count, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 计算清晰度（拉普拉斯方差）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['sharpness'].append(sharpness)
            
            # 计算亮度
            brightness = np.mean(gray)
            metrics['brightness'].append(brightness)
            
            # 检测面部置信度
            face_locations = face_recognition.face_locations(frame)
            face_confidence = len(face_locations) > 0
            metrics['face_confidence'].append(float(face_confidence))
        
        cap.release()
        
        # 计算平均值
        quality_scores = {
            'sharpness': np.mean(metrics['sharpness']),
            'brightness': np.mean(metrics['brightness']),
            'face_detection_rate': np.mean(metrics['face_confidence'])
        }
        
        return quality_scores
    
    def check_audio_quality(self, video_path: str) -> Dict[str, float]:
        """检查音频质量"""
        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                return {'has_audio': False, 'snr': 0, 'energy': 0}
            
            # 提取音频
            audio_array = video.audio.to_soundarray(fps=16000)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            video.close()
            
            # 计算信噪比（简化版本）
            signal_power = np.mean(audio_array**2)
            noise_power = np.var(audio_array)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # 计算能量
            energy = np.sqrt(signal_power)
            
            return {
                'has_audio': True,
                'snr': snr,
                'energy': energy
            }
            
        except Exception as e:
            logger.error(f"Error checking audio quality: {e}")
            return {'has_audio': False, 'snr': 0, 'energy': 0}
    
    def filter_dataset(self, 
                      video_dir: str,
                      output_list: str,
                      min_quality_threshold: Dict[str, float] = None) -> List[str]:
        """根据质量筛选数据集"""
        if min_quality_threshold is None:
            min_quality_threshold = {
                'sharpness': 20.0,
                'brightness': 50.0,
                'face_detection_rate': 0.8,
                'snr': 10.0,
                'energy': 0.01
            }
        
        video_dir = Path(video_dir)
        video_files = list(video_dir.rglob("*.mp4"))
        
        high_quality_videos = []
        quality_report = []
        
        for video_file in tqdm(video_files, desc="Checking quality"):
            # 检查视频质量
            video_quality = self.check_video_quality(str(video_file))
            audio_quality = self.check_audio_quality(str(video_file))
            
            # 合并质量指标
            quality_metrics = {**video_quality, **audio_quality}
            quality_metrics['path'] = str(video_file)
            
            # 判断是否通过质量检查
            passed = all(
                quality_metrics.get(metric, 0) >= threshold
                for metric, threshold in min_quality_threshold.items()
            )
            
            quality_metrics['passed'] = passed
            quality_report.append(quality_metrics)
            
            if passed:
                high_quality_videos.append(str(video_file))
        
        # 保存质量报告
        report_path = Path(output_list).parent / "quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        # 保存高质量视频列表
        with open(output_list, 'w') as f:
            for video in high_quality_videos:
                f.write(f"{video}\n")
        
        logger.info(f"Found {len(high_quality_videos)}/{len(video_files)} high quality videos")
        return high_quality_videos

# ==================== 数据集拆分 ====================
def split_dataset(video_list: str, 
                 output_dir: str,
                 train_ratio: float = 0.9,
                 val_ratio: float = 0.05,
                 test_ratio: float = 0.05,
                 seed: int = 42):
    """将数据集拆分为训练/验证/测试集"""
    np.random.seed(seed)
    
    # 读取视频列表
    with open(video_list, 'r') as f:
        videos = [line.strip() for line in f.readlines()]
    
    # 随机打乱
    np.random.shuffle(videos)
    
    # 计算拆分点
    n_total = len(videos)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 拆分
    train_videos = videos[:n_train]
    val_videos = videos[n_train:n_train + n_val]
    test_videos = videos[n_train + n_val:]
    
    # 保存拆分结果
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    splits = {
        'train': train_videos,
        'val': val_videos,
        'test': test_videos
    }
    
    for split_name, split_videos in splits.items():
        split_file = output_path / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for video in split_videos:
                f.write(f"{video}\n")
        
        logger.info(f"{split_name}: {len(split_videos)} videos")
    
    # 保存拆分统计
    stats = {
        'total': n_total,
        'train': len(train_videos),
        'val': len(val_videos),
        'test': len(test_videos),
        'seed': seed
    }
    
    with open(output_path / "split_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="Sonic数据准备工具")
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 下载命令
    download_parser = subparsers.add_parser('download', help='下载数据集')
    download_parser.add_argument('--dataset', choices=['voxceleb2', 'celebv_hq', 'vfhq', 'all'], 
                                default='all', help='要下载的数据集')
    download_parser.add_argument('--root_dir', default='./datasets', help='数据集根目录')
    
    # 预处理命令
    preprocess_parser = subparsers.add_parser('preprocess', help='预处理视频')
    preprocess_parser.add_argument('--input_dir', required=True, help='输入视频目录')
    preprocess_parser.add_argument('--output_dir', required=True, help='输出视频目录')
    preprocess_parser.add_argument('--resolution', type=int, default=512, help='目标分辨率')
    preprocess_parser.add_argument('--fps', type=int, default=25, help='目标帧率')
    preprocess_parser.add_argument('--workers', type=int, default=4, help='并行处理数')
    
    # 质量检查命令
    quality_parser = subparsers.add_parser('quality', help='检查数据质量')
    quality_parser.add_argument('--video_dir', required=True, help='视频目录')
    quality_parser.add_argument('--output_list', required=True, help='高质量视频列表输出路径')
    
    # 数据集拆分命令
    split_parser = subparsers.add_parser('split', help='拆分数据集')
    split_parser.add_argument('--video_list', required=True, help='视频列表文件')
    split_parser.add_argument('--output_dir', required=True, help='输出目录')
    split_parser.add_argument('--train_ratio', type=float, default=0.9, help='训练集比例')
    split_parser.add_argument('--val_ratio', type=float, default=0.05, help='验证集比例')
    split_parser.add_argument('--test_ratio', type=float, default=0.05, help='测试集比例')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        downloader = DatasetDownloader(args.root_dir)
        
        if args.dataset in ['voxceleb2', 'all']:
            downloader.download_voxceleb2(username='', password='')
        
        if args.dataset in ['celebv_hq', 'all']:
            downloader.download_celebv_hq()
        
        if args.dataset in ['vfhq', 'all']:
            downloader.download_vfhq()
    
    elif args.command == 'preprocess':
        preprocessor = VideoPreprocessor(
            target_fps=args.fps,
            target_resolution=args.resolution
        )
        preprocessor.process_dataset(
            args.input_dir,
            args.output_dir,
            num_workers=args.workers
        )
    
    elif args.command == 'quality':
        checker = DataQualityChecker()
        checker.filter_dataset(args.video_dir, args.output_list)
    
    elif args.command == 'split':
        split_dataset(
            args.video_list,
            args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
