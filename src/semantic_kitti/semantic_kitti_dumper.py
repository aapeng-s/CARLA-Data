import os
import cv2
import numpy as np
from dataclasses import dataclass

from packages.carla1s.actors import RgbCamera, DepthCamera, SemanticLidar
from packages.carla1s.tf import Point, CoordConverter

from ..dataset_dumper import DatasetDumper

class SemanticKittiDumper(DatasetDumper):
    
    @dataclass
    class SemanticLidarTargetPair(DatasetDumper.SensorTargetPair):
        label_path: str
        
    @dataclass
    class TimestampTargetPair(DatasetDumper.SensorTargetPair):
        pass
    
    @dataclass
    class ImageTargetPair(DatasetDumper.SensorTargetPair):
        pass
    
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
        for bind in self.binds:
            if isinstance(bind, self.SemanticLidarTargetPair):
                self._promises.append(self.thread_pool.submit(self._dump_semantic_lidar, bind))
            elif isinstance(bind, self.ImageTargetPair):
                self._promises.append(self.thread_pool.submit(self._dump_image, bind))

        return self
    
    def bind_camera(self, sensor: Sensor, data_path: str) -> 'DatasetDumper':
        self._binds.append(self.ImageTargetPair(sensor, data_path))
        return self

    def bind_semantic_lidar(self, sensor: Sensor, data_path: str, label_path: str) -> 'DatasetDumper':
        self._binds.append(self.SemanticLidarTargetPair(sensor, data_path, label_path))
        return self

    def _setup_content_folder(self):
        """创建内容文件夹."""
        for bind in self.binds:
            os.makedirs(os.path.join(self.current_sequence_path, bind.data_path))
            self.logger.info(f"Created data path: {os.path.join(self.current_sequence_path, bind.data_path)}")
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

