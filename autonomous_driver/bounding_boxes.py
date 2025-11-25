import carla
import json
import random
import queue
import pygame
import argparse
import logging
import numpy as np

from carla.command import SpawnActor, SetAutopilot, FutureActor, DestroyActor
from pygame.locals import K_ESCAPE
from pygame.locals import K_2
from pygame.locals import K_3
from pygame.locals import K_r

from util import (
    weather_dict,
    EDGES,
    SEMANTIC_MAP,
    build_projection_matrix,
    get_image_point,
    point_in_canvas,
    decode_instance_segmentation,
    bbox_2d_for_actor,
    bbox_3d_for_actor,
    visualize_2d_bboxes,
    visualize_3d_bboxes,
    calculate_relative_velocity,
    vehicle_light_state_to_dict,
)

def main():

    argparser = argparse.ArgumentParser(
        description='CARLA bounding boxes')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        "-m", "--map",
        metavar="M",
        help="Name of map to load in world",
    )
    argparser.add_argument(
        "--weather",
        metavar="W",
        default="ClearNoon",
        help="Carla weather preset"
    )
    argparser.add_argument(
        "-n", "--number-of-vehicles",
        metavar="N",
        default=30,
        type=int,
        help="Number of vehicles (default: 30)",
    )
    argparser.add_argument(
        '-d', '--distance',
        metavar='D',
        default=50,
        type=int,
        help='Actor distance threshold')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

    pygame.init()

    # State variables
    record = False
    display_3d = False
    run_simulation = True

    clock = pygame.time.Clock()
    pygame.display.set_caption("Bounding Box Visualization")
    display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
    display.fill((0,0,0))
    pygame.display.flip()

    # Connect to the CARLA server and get the world object
    client = carla.Client(args.host, args.port)
    world  = client.get_world()

    if args.map:
        if args.map not in client.get_available_maps():
            raise ValueError("Could not find any map with the given name")
        world = client.load_world(args.map)

    if args.weather not in weather_dict:
        raise ValueError("Unknown Carla weather preset")
    world.set_weather(weather_dict[args.weather])

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Set the traffic manager to Synchronous mode
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    bp_lib = world.get_blueprint_library()

    # Get the map spawn points
    spawn_points = world.get_map().get_spawn_points()

    # spawn vehicle
    vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
    ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

    # spawn RGB camera
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(args.width))
    camera_bp.set_attribute('image_size_y', str(args.height))
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

    # spawn instance segmentation camera
    inst_camera_bp = bp_lib.find('sensor.camera.instance_segmentation')
    inst_camera_bp.set_attribute('image_size_x', str(args.width))
    inst_camera_bp.set_attribute('image_size_y', str(args.height))
    camera_init_trans = carla.Transform(carla.Location(z=2))
    inst_camera = world.spawn_actor(inst_camera_bp, camera_init_trans, attach_to=ego_vehicle)

    ego_vehicle.set_autopilot(True)

    # Add some traffic
    num_spawn_points = len(spawn_points)

    if args.number_of_vehicles < num_spawn_points:
        random.shuffle(spawn_points)
    elif args.number_of_vehicles > num_spawn_points:
        msg = f"Requested {args.number_of_vehicles} vehicles but could only find {num_spawn_points} spawn points"
        logging.warning(msg)
        args.number_of_vehicles = num_spawn_points

    npcs = []
    for n, tranform in enumerate(spawn_points):
        if n >= args.number_of_vehicles:
            break
        vehicle_bp = random.choice(bp_lib.filter('vehicle'))
        if vehicle_bp.has_attribute("color"):
            color = random.choice(vehicle_bp.get_attribute("color").recommended_values)
            vehicle_bp.set_attribute("color", color)
        if vehicle_bp.has_attribute("driver_id"):
            driver_id = random.choice(vehicle_bp.get_attribute("driver_id").recommended_values)
            vehicle_bp.set_attribute("driver_id", driver_id)
        vehicle_bp.set_attribute("role_name", "autopilot")

        npc = world.try_spawn_actor(vehicle_bp, tranform)
        if npc:
            npc.set_autopilot(True)
            npcs.append(npc)
        
    logging.info(f"Spawned {len(npcs)} vehicles")

    # Create queues to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    inst_queue = queue.Queue()
    inst_camera.listen(inst_queue.put)

    try:
        while run_simulation:
            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if event.key == K_r:
                        record = True
                    if event.key == K_2:
                        display_3d = False
                    if event.key == K_3:
                        display_3d = True
                    if event.key == K_ESCAPE:
                        run_simulation = False
                if event.type == pygame.QUIT:
                    run_simulation = False

            world.tick()
            snapshot = world.get_snapshot()

            json_frame_data = {
                'frame_id': snapshot.frame,
                'timestamp': snapshot.timestamp.elapsed_seconds,
                'objects': [] 
            }

            image = image_queue.get()
            img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

            if record:
                image.save_to_disk('_out/%08d' % image.frame)

            inst_seg_image = inst_queue.get()
            inst_seg = np.reshape(np.copy(inst_seg_image.raw_data), (inst_seg_image.height, inst_seg_image.width, 4))

            # Decode instance segmentation image
            semantic_labels, actor_ids = decode_instance_segmentation(inst_seg)

            # Empty list to collect bounding boxes for this frame
            frame_bboxes = []

            # Loop through the NPCs in the simulation
            for npc in world.get_actors().filter('*vehicle*'):

                # Filter out the ego vehicle
                if npc.id !=ego_vehicle.id:

                    npc_bbox = npc.bounding_box
                    dist = npc.get_transform().location.distance(ego_vehicle.get_transform().location)

                    # Filter for the vehicles within 50m
                    if dist < args.distance:

                        # Limit to vehicles in front of the camera
                        forward_vec = camera.get_transform().get_forward_vector()
                        inter_vehicle_vec = npc.get_transform().location - camera.get_transform().location

                        if forward_vec.dot(inter_vehicle_vec) > 0:
                            
                            # Generate 2D and 2D bounding boxes for each actor
                            npc_bbox_2d = bbox_2d_for_actor(npc, actor_ids, semantic_labels)
                            npc_bbox_3d = bbox_3d_for_actor(npc, ego_vehicle, camera_bp, camera)

                            frame_bboxes.append({'3d': npc_bbox_3d, '2d': npc_bbox_2d})

                            json_frame_data['objects'].append({
                                'id': npc.id,
                                'class': SEMANTIC_MAP[npc.semantic_tags[0]][0],
                                'blueprint_id': npc.type_id,
                                'velocity': calculate_relative_velocity(npc, ego_vehicle),
                                'bbox_3d': npc_bbox_3d['bbox_3d'],
                                'bbox_2d': {
                                    'xmin': int(npc_bbox_2d['bbox_2d'][0]),
                                    'ymin': int(npc_bbox_2d['bbox_2d'][1]),
                                    'xmax': int(npc_bbox_2d['bbox_2d'][2]),
                                    'ymax': int(npc_bbox_2d['bbox_2d'][3]),
                                } if npc_bbox_2d else None,
                                'light_state': vehicle_light_state_to_dict(npc)

                            })

            # Draw the scene in Pygame
            display.fill((0,0,0))
            if display_3d:
                visualize_3d_bboxes(display, img, frame_bboxes)
            else:
                visualize_2d_bboxes(display, img, frame_bboxes)
            pygame.display.flip()
            clock.tick(30)  # 30 FPS              
            if record:
                with open(f"_out/{snapshot.frame}.json", 'w') as f:
                    json.dump(json_frame_data, f)

    except KeyboardInterrupt:
        pass
    finally:
        
        ego_vehicle.destroy()
        camera.stop()
        camera.destroy()
        inst_camera.stop()
        inst_camera.destroy()
        for npc in npcs:
            npc.set_autopilot(False)
            npc.destroy()

        world.tick()

        # Set up the simulator in synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = False # Disables synchronous mode
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        # Set the traffic manager to Synchronous mode
        traffic_manager.set_synchronous_mode(False)

        pygame.quit()

        print('\ndone.')


if __name__ == '__main__':
    print('Bounding boxes script instructions:')
    print('R    : toggle recording images as PNG and bounding boxes as JSON')
    print('3    : view the bounding boxes in 3D')
    print('2    : view the bounding boxes in 2D')
    print('ESC  : quit')
    main()
