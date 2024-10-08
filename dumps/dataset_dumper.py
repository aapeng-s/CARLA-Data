from abc import ABC, abstractmethod
from typing import Optional
from concurrent.futures import ThreadPoolExecutor


class DatasetDumper(ABC):
    """数据集导出器抽象类"""
    
    def __init__(self, root_path: str, max_workers: int = 4) -> None:
        self._root_path = root_path
        self._current_sequence_name: Optional[str] = None
        self._current_frame_count: int = 0
        self._dump_thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
    def __del__(self):
        self._dump_thread_pool.shutdown(wait=True)

    @property
    def root_path(self) -> str:
        """根目录路径"""
        return self._root_path
    
    @property
    def dump_thread_pool(self) -> ThreadPoolExecutor:
        """IO 写入操作的线程池"""
        return self._dump_thread_pool
    
    @property
    def current_sequence_name(self) -> Optional[str]:
        """当前序列名称"""
        return self._current_sequence_name
    
    @property
    def current_frame_count(self) -> int:
        """当前帧计数"""
        return self._current_frame_count

    @property
    @abstractmethod
    def current_frame_name(self) -> str:
        """当前帧名称, 一般由帧计数器处理得到"""
        pass
    
    @abstractmethod
    def build_sequence(self, sequence_name: str):
        """构建序列
        
        通常一个序列代表一个场景, 一个场景包含多个帧
        """
        pass
    
    @abstractmethod
    def build_misc(self):
        """构建杂项数据
        
        杂项数据通常包括相机参数, 传感器参数等配置性数据
        """
        pass
    
    @abstractmethod
    def build_frame(self):
        """构建帧
        
        帧是数据的基本单位, 通常包含多个传感器数据
        """
        pass
