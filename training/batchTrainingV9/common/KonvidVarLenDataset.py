import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from torchvision import transforms as T
import random
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

class KonvidVarLenDataset(Dataset):
    """
    Enhanced dataset for KoNViD-1k with performance optimizations:
    - Frame sampling strategies for long videos
    - Optional caching for faster loading
    - Prefetching mechanism
    """
    def __init__(self, video_dir, labels_csv, id_column, label_columns,
                 sequence_length=10, resize_dim=(224, 224), csv_sep=",",
                 sampling_strategy='uniform', cache_size=100,
                 enable_prefetch=True, num_prefetch_workers=2,
                 multiprocessing_context=None):
        """
        Args:
            video_dir: Directory containing video files
            labels_csv: Path to CSV file with labels
            id_column: Column name in CSV containing video IDs
            label_columns: List of column names to use as labels
            sequence_length: Number of frames to extract from each video
            resize_dim: Dimensions to resize frames to
            csv_sep: CSV separator character
            sampling_strategy: How to sample frames ('uniform', 'random', or 'first')
            cache_size: Number of videos to cache in memory (0 to disable)
            enable_prefetch: Whether to prefetch videos in background
            num_prefetch_workers: Number of worker threads for prefetching
            multiprocessing_context: If provided, indicates this dataset will be used with multiprocessing
        """
        self.video_dir = video_dir
        self.sequence_length = sequence_length
        self.label_columns = label_columns
        self.sampling_strategy = sampling_strategy
        self.resize = T.Resize(resize_dim, antialias=True)
        # ImageNet normalization - optional, can be disabled if causing issues
        self.use_normalize = True
        try:
            self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        except Exception as e:
            logger.warning(f"Could not create normalization transform: {e}")
            self.use_normalize = False

        # Read and process labels
        df = pd.read_csv(labels_csv, sep=csv_sep, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        self.label_map = {
            str(row[id_column]): row[label_columns].to_dict()
            for _, row in df.iterrows()
        }

        # Min-max normalization parameters
        self.label_mins = {}
        self.label_maxs = {}
        for col in label_columns:
            self.label_mins[col] = df[col].min()
            self.label_maxs[col] = df[col].max()

        # Find all valid video files
        self.samples = []
        for fname in os.listdir(video_dir):
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_id = os.path.splitext(fname)[0].split("_")[0]
                if video_id in self.label_map:
                    self.samples.append((os.path.join(video_dir, fname), video_id))

        # Setup caching if enabled
        self.cache_size = cache_size
        self.cache = OrderedDict()
        if cache_size > 0:
            logger.info(f"Video caching enabled with size {cache_size}")

        # Detect if we're in a multiprocessing context
        self.is_multiprocessing = multiprocessing_context is not None

        # Setup prefetching if enabled (disable when using multiprocessing)
        self.enable_prefetch = enable_prefetch and not self.is_multiprocessing
        self.num_prefetch_workers = num_prefetch_workers

        if self.enable_prefetch:
            self.prefetch_queue = {}
            self.prefetch_lock = threading.Lock()
            self.prefetch_executor = ThreadPoolExecutor(max_workers=num_prefetch_workers)
            self.prefetch_indices = set()
            logger.info(f"Prefetching enabled with {num_prefetch_workers} workers")
        else:
            # Create dummy attributes that won't cause pickling issues
            self.prefetch_queue = None
            self.prefetch_lock = None
            self.prefetch_executor = None
            self.prefetch_indices = None
            if enable_prefetch and self.is_multiprocessing:
                logger.info("Prefetching disabled because multiprocessing is being used")

    def __len__(self):
        return len(self.samples)

    def _sample_frame_indices(self, total_frames):
        """Sample frame indices based on the chosen strategy"""
        if total_frames <= self.sequence_length:
            # If video has fewer frames than needed, return all available indices
            return list(range(total_frames))

        if self.sampling_strategy == 'uniform':
            # Uniformly sample frames across the video
            return [int(i * total_frames / self.sequence_length) for i in range(self.sequence_length)]
        elif self.sampling_strategy == 'random':
            # Randomly sample frames
            return sorted(random.sample(range(total_frames), self.sequence_length))
        elif self.sampling_strategy == 'first':
            # Take the first N frames
            return list(range(min(total_frames, self.sequence_length)))
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _get_video_tensor(self, path):
        """Load and process video frames from file"""
        # Check if the video is in the cache
        if self.cache_size > 0 and path in self.cache:
            return self.cache[path]

        try:
            ctx = cpu(0)
            vr = VideoReader(path, ctx=ctx)
            total_frames = len(vr)

            # Sample frame indices based on strategy
            frame_indices = self._sample_frame_indices(total_frames)
            n = len(frame_indices)

            # Get the selected frames
            frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)

            # Transform frames
            transformed_frames = []
            for f in frames:
                t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0  # (C, H, W)
                t_resized = self.resize(t)  # (C, H, W)
                # Apply normalization if enabled
                if self.use_normalize:
                    try:
                        t_normalized = self.normalize(t_resized)
                        transformed_frames.append(t_normalized)
                    except Exception as e:
                        logger.warning(f"Normalization failed, using unnormalized frame: {e}")
                        transformed_frames.append(t_resized)
                else:
                    transformed_frames.append(t_resized)

            # Padding if needed
            if n < self.sequence_length:
                pad = [torch.zeros_like(transformed_frames[0]) for _ in range(self.sequence_length - n)]
                transformed_frames.extend(pad)

            # Stack frames into tensor
            video_tensor = torch.stack(transformed_frames, dim=0)  # (T, C, H, W)

            # Add to cache if caching is enabled
            if self.cache_size > 0:
                # If cache is full, remove the oldest item
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)  # Remove oldest item (FIFO)
                self.cache[path] = video_tensor

            return video_tensor

        except Exception as e:
            logger.error(f"Error loading video {path}: {e}")
            # Return a tensor of zeros as fallback
            return torch.zeros((self.sequence_length, 3, *self.resize.size))

    def _prefetch(self, index):
        """Prefetch a video in the background"""
        path, _ = self.samples[index]
        try:
            # Use a thread-safe approach to get the video tensor
            video_tensor = self._get_video_tensor(path)
            with self.prefetch_lock:
                self.prefetch_queue[index] = video_tensor
                # Successfully prefetched, can remove from pending indices
                if index in self.prefetch_indices:
                    self.prefetch_indices.remove(index)
        except Exception as e:
            logger.error(f"Error prefetching video {path}: {e}")
            # Remove from pending indices even if failed
            with self.prefetch_lock:
                if index in self.prefetch_indices:
                    self.prefetch_indices.remove(index)

    def prefetch_batch(self, indices):
        """Prefetch a batch of videos"""
        if not self.enable_prefetch:
            return

        for idx in indices:
            if idx not in self.prefetch_indices and idx < len(self.samples):
                self.prefetch_indices.add(idx)
                self.prefetch_executor.submit(self._prefetch, idx)

    def __getitem__(self, index):
        # Check if the video is already prefetched
        if self.enable_prefetch:
            with self.prefetch_lock:
                if index in self.prefetch_queue:
                    video_tensor = self.prefetch_queue.pop(index)
                    # No need to remove from prefetch_indices as it's already done in _prefetch
                else:
                    # If not prefetched, load it directly
                    path, _ = self.samples[index]
                    video_tensor = self._get_video_tensor(path)
        else:
            # No prefetching, load directly
            path, _ = self.samples[index]
            video_tensor = self._get_video_tensor(path)

        # Get and normalize labels
        _, video_id = self.samples[index]
        label_dict = self.label_map[video_id]
        labels = []

        for col in self.label_columns:
            value = label_dict[col]
            min_val = self.label_mins[col]
            max_val = self.label_maxs[col]
            range_val = max_val - min_val if max_val != min_val else 1.0
            norm_value = (value - min_val) / range_val
            labels.append(norm_value)

        labels = torch.tensor(labels, dtype=torch.float32)

        # Prefetch next items for future batches
        if self.enable_prefetch:
            next_indices = [index + i for i in range(1, 5) if index + i < len(self.samples)]
            self.prefetch_batch(next_indices)

        return video_tensor, labels

    def denormalize(self, preds):
        """
        Denormalize predictions back to original ranges.

        Args:
            preds (Tensor): predictions (N x num_labels)

        Returns:
            Tensor: denormalized predictions (N x num_labels)
        """
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)  # (1, num_labels)

        denorm_preds = []
        for i, col in enumerate(self.label_columns):
            min_val = self.label_mins[col]
            max_val = self.label_maxs[col]
            range_val = max_val - min_val if max_val != min_val else 1.0
            real_value = preds[:, i] * range_val + min_val
            denorm_preds.append(real_value)

        return torch.stack(denorm_preds, dim=1)  # (N, num_labels)
    
    def get_label_ranges(self):
        return self.label_mins, self.label_maxs
