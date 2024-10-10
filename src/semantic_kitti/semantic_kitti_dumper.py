import os
import cv2
from concurrent.futures import as_completed

from packages.carla1s.actors import RgbCamera, DepthCamera, SemanticLidar

from ..dataset_dumper import DatasetDumper

class SemanticKittiDumper(DatasetDumper):
    
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
            if isinstance(bind, DatasetDumper.SemanticLidarTargetPair):
                self._promises.append(self.thread_pool.submit(self._dump_semantic_lidar, bind))
            else:
                self._promises.append(self.thread_pool.submit(self._dump_image, bind))

        return self

    def _setup_content_folder(self):
        """创建内容文件夹."""
        for bind in self.binds:
            os.makedirs(os.path.join(self.current_sequence_path, bind.data_path))
            self.logger.info(f"Created data path: {os.path.join(self.current_sequence_path, bind.data_path)}")
            # 处理特殊项
            if isinstance(bind, DatasetDumper.SemanticLidarTargetPair):
                os.makedirs(os.path.join(self.current_sequence_path, bind.label_path))
                self.logger.info(f"Created label path: {os.path.join(self.current_sequence_path, bind.label_path)}")

    def _setup_calib_file(self):
        """创建标定文件."""
        pass

    def _dump_image(self, bind: DatasetDumper.SensorTargetPair):
        # 阻塞等待传感器更新
        bind.sensor.on_data_ready.wait()
        # 储存数据
        file_name = f"{self.current_frame_name}.png"
        path = os.path.join(self.current_sequence_path, bind.data_path, file_name)
        cv2.imwrite(path, bind.sensor.data.data)
        # 打印日志
        self.logger.debug(f"[frame={bind.sensor.data.frame}] Dumped image to {path}")

    def _dump_semantic_lidar(self, bind: DatasetDumper.SemanticLidarTargetPair):
        # 阻塞等待传感器更新
        bind.sensor.on_data_ready.wait()
        # 打印日志
        self.logger.debug(f"[frame={bind.sensor.data.frame}] Dumped pointcloud to {bind.data_path}")

