import logging

from packages.carla1s import CarlaContext, ManualExecutor
from packages.carla1s.actors import Vehicle, RgbCamera, DepthCamera, SemanticLidar
from packages.carla1s.tf import Transform

from src.semantic_kitti import SemanticKittiDumper


def main(*, fps: int = 20):   

    with CarlaContext(host='10.0.0.110', log_level=logging.DEBUG) as cc, ManualExecutor(cc, fixed_delta_seconds=1/fps) as exe:        
        ego_vehicle: Vehicle = (cc.actor_factory
            .create(Vehicle, from_blueprint='vehicle.tesla.model3')
            .with_name("ego_vehicle")
            .with_transform(cc.get_spawn_point(0))
            .build())
        
        cam_0_tf = Transform(x=0.30, y=0.00, z=1.70)
        cam_1_tf = Transform(x=0.30, y=0.50, z=1.70)
        semantic_lidar_tf = Transform(x=0.00, y=0.00, z=1.80)

        cam_0_rgb: RgbCamera = (cc.actor_factory
            .create(RgbCamera)
            .with_name("cam_0_rgb")
            .with_transform(cam_0_tf)
            .with_parent(ego_vehicle)
            .build())
        cam_0_depth: DepthCamera = (cc.actor_factory
            .create(DepthCamera)
            .with_name("cam_0_depth")
            .with_transform(cam_0_tf)
            .with_parent(ego_vehicle)
            .build())
            
        cam_1_rgb: RgbCamera = (cc.actor_factory
            .create(RgbCamera)
            .with_name("cam_1_rgb")
            .with_transform(cam_1_tf)
            .with_parent(ego_vehicle)
            .build())
        cam_1_depth: DepthCamera = (cc.actor_factory
            .create(DepthCamera)
            .with_name("cam_1_depth")
            .with_transform(cam_1_tf)
            .with_parent(ego_vehicle)
            .build())
            
        semantic_lidar: SemanticLidar = (cc.actor_factory
            .create(SemanticLidar)
            .with_name("semantic_lidar")
            .with_transform(semantic_lidar_tf)
            .with_parent(ego_vehicle)
            .with_attributes(rotation_frequency=fps)
            .build())
            
        cc.all_actors_spawn().all_sensors_listen()
        exe.wait_ticks(1)
        
        ego_vehicle.set_autopilot(True)
        exe.wait_ticks(1)
        
        # SETUP DUMPER
        dumper = SemanticKittiDumper('/home/isleep/Documents/temp_kitti')
        dumper.bind_camera(cam_0_depth, "image_0")
        dumper.bind_camera(cam_1_depth, "image_1")
        dumper.bind_camera(cam_0_rgb, "image_2")
        dumper.bind_camera(cam_1_rgb, "image_3")
        dumper.bind_semantic_lidar(semantic_lidar, "velodyne", "labels")
        dumper.bind_timestamp(cam_0_rgb, "times.txt")
        dumper.bind_pose(cam_0_rgb, "poses.txt")
        dumper.bind_calib(semantic_lidar, "calib.txt")

        dumper.create_sequence()
        
        
        for i in range(3):
            exe.wait_ticks(1)
            dumper.create_frame().join()


if __name__ == "__main__":
    main()
