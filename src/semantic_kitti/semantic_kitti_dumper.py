import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from packages.carla1s.actors import Sensor
from packages.carla1s.tf import Point, CoordConverter, Transform

from ..dataset_dumper import DatasetDumper

class SemanticKittiDumper(DatasetDumper):
    
    @dataclass
    class SemanticLidarTargetPair(DatasetDumper.SensorTargetPair):
        label_path: str
        
    @dataclass
    class TimestampTargetPair(DatasetDumper.SensorTargetPair):
        pass
    
    @dataclass
    class PoseTargetPair(DatasetDumper.SensorTargetPair):
        pass
    
    @dataclass
    class ImageTargetPair(DatasetDumper.SensorTargetPair):
        pass
    
    def __init__(self, root_path: str, max_workers: int = 3):
        super().__init__(root_path, max_workers)
        self._timestamp_offset: Optional[float] = None
        self._pose_offset: Optional[Transform] = None
    
    @property
    def current_frame_name(self) -> str:
        return f'{self._current_frame_count:06d}'

    def create_sequence(self, name: str = None):
        super().create_sequence(name)
        self._setup_content_folder()

    def create_frame(self) -> 'SemanticKittiDumper':
        # 遍历所有的 bind, 创建 dump 任务
        self._current_frame_count += 1
        self._promises = []
        
        # 处理第一帧的特殊情况, 标记 offset 为 None
        if self._current_frame_count == 1:
            self._timestamp_offset = None
            self._pose_offset = None
        for bind in self.binds:
            if isinstance(bind, self.SemanticLidarTargetPair):
                self._promises.append(self.thread_pool.submit(self._dump_semantic_lidar, bind))
            elif isinstance(bind, self.ImageTargetPair):
                self._promises.append(self.thread_pool.submit(self._dump_image, bind))
            elif isinstance(bind, self.TimestampTargetPair):
                self._promises.append(self.thread_pool.submit(self._dump_timestamp, bind))
            elif isinstance(bind, self.PoseTargetPair):
                self._promises.append(self.thread_pool.submit(self._dump_pose, bind))

        return self
    
    def bind_camera(self, sensor: Sensor, data_path: str) -> 'DatasetDumper':
        self._binds.append(self.ImageTargetPair(sensor, data_path))
        return self

    def bind_semantic_lidar(self, sensor: Sensor, data_path: str, label_path: str) -> 'DatasetDumper':
        self._binds.append(self.SemanticLidarTargetPair(sensor, data_path, label_path))
        return self

    def bind_timestamp(self, sensor: Sensor, path: str):
        if os.path.splitext(path)[1] == '':
            raise ValueError(f"Path {path} is a folder, not a file.")
        self.binds.append(self.TimestampTargetPair(sensor, path))
        
    def bind_pose(self, sensor: Sensor, path: str):
        if os.path.splitext(path)[1] == '':
            raise ValueError(f"Path {path} is a folder, not a file.")
        self.binds.append(self.PoseTargetPair(sensor, path))

    def _setup_content_folder(self):
        """创建内容文件夹."""
        for bind in self.binds:
            # 如果以扩展名结尾则创建文件，否则创建目录
            if os.path.splitext(bind.data_path)[1] == '':
                os.makedirs(os.path.join(self.current_sequence_path, bind.data_path))
                self.logger.info(f"Created folder at: {os.path.join(self.current_sequence_path, bind.data_path)}")
            else:
                with open(os.path.join(self.current_sequence_path, bind.data_path), 'w') as f:
                    f.write('')
                    self.logger.info(f"Created file at: {os.path.join(self.current_sequence_path, bind.data_path)}")
            # 处理特殊项
            if isinstance(bind, self.SemanticLidarTargetPair):
                os.makedirs(os.path.join(self.current_sequence_path, bind.label_path))
                self.logger.info(f"Created label path: {os.path.join(self.current_sequence_path, bind.label_path)}")

    def _setup_calib_file(self):
        """创建标定文件."""
        pass

    def _dump_image(self, bind: ImageTargetPair):
        # 阻塞等待传感器更新
        bind.sensor.on_data_ready.wait()
        # 储存数据
        file_name = f"{self.current_frame_name}.png"
        path = os.path.join(self.current_sequence_path, bind.data_path, file_name)
        cv2.imwrite(path, bind.sensor.data.content)
        # 打印日志
        self.logger.debug(f"[frame={bind.sensor.data.frame}] Dumped image to {path}")

    def _dump_semantic_lidar(self, bind: SemanticLidarTargetPair):
        # 阻塞等待传感器更新
        bind.sensor.on_data_ready.wait()
        
        # 准备储存路径
        file_name = f"{self.current_frame_name}"
        path_data = os.path.join(self.current_sequence_path, bind.data_path, file_name + '.bin')
        path_label = os.path.join(self.current_sequence_path, bind.label_path, file_name + '.label')
        
        # 处理点云
        points = [Point(x=x, y=y, z=z) for x, y, z in bind.sensor.data.content[:, :3]]
        points = CoordConverter.from_system(*points).apply_transform(CoordConverter.TF_TO_KITTI).get_list()
        points = np.array([[p.x, p.y, p.z, 1.0] for p in points], dtype=np.float32)
                
        # 处理标注
        seg = bind.sensor.data.content[:, 3]
        id = bind.sensor.data.content[:, 4]
        labels = np.column_stack((seg.astype(np.uint16), id.astype(np.uint16)))
        
        # 储存数据
        points.tofile(path_data)
        labels.tofile(path_label)
        
        # 打印日志
        self.logger.debug(f"[frame={bind.sensor.data.frame}] Dumped pointcloud to {path_data}")
        self.logger.debug(f"[frame={bind.sensor.data.frame}] Dumped labels to {path_label}")

    def _dump_timestamp(self, bind: DatasetDumper.SensorTargetPair):
        """导出时间戳, 以秒为单位, 使用科学计数法, 保留小数点后 6 位.

        Args:
            bind (DatasetDumper.SensorTargetPair): 参考的传感器绑定
        """
        bind.sensor.on_data_ready.wait()
        if self._timestamp_offset is None:
            self._timestamp_offset = bind.sensor.data.timestamp
        timestamp = bind.sensor.data.timestamp - self._timestamp_offset
        with open(os.path.join(self.current_sequence_path, bind.data_path), 'a') as f:
            f.write(f"{timestamp:.6e}\n")
            
        self.logger.debug(f"[frame={bind.sensor.data.frame}] Dumped timestamp to {os.path.join(self.current_sequence_path, bind.data_path)}, value: {timestamp:.6e}")

    def _dump_pose(self, bind: PoseTargetPair):
        """导出位姿数据, 是 3x4 的变换矩阵, 表示当前帧参考传感器到初始帧参考传感器的位姿变换.

        Args:
            bind (PoseTargetPair): 参考的传感器绑定
        """
        bind.sensor.on_data_ready.wait()
        
        # 准备储存路径
        path = os.path.join(self.current_sequence_path, bind.data_path)
        
        # 获取位姿数据并转换为
        pose = bind.sensor.data.transform
        
        # 如果 offset 未设置, 则设置为当前帧的位姿
        if self._pose_offset is None:
            self._pose_offset = pose
        
        # 计算当前帧的位姿相对初始帧的位姿
        relative_pose = (CoordConverter
                         .from_system(pose)
                         .apply_transform(self._pose_offset)
                         .apply_transform(CoordConverter.TF_TO_KITTI)
                         .get_single())
        
        # 将位姿矩阵转换为 3x4 的变换矩阵
        pose_matrix = relative_pose.matrix[:3, :]
        
        # 横向展开, 表示为 1x12 的行向量, 并处理为小数点后 6 位的科学计数法表示, 以空格分隔
        pose_matrix = pose_matrix.flatten()
        pose_matrix = [f"{value:.6e}" for value in pose_matrix]
        pose_matrix = ' '.join(pose_matrix)

        # 保存到文件
        with open(path, 'a') as f:
            f.write(f"{pose_matrix}\n")
        self.logger.debug(f"[frame={bind.sensor.data.frame}] Dumped pose to {path}")
