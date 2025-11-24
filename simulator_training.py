#1/usr/bin/env python

"""Script to generate world and gather training data for autonomous driving"""

import carla
from carla.command import SpawnActor, SetAutopilot, FutureActor, DestroyActor

import argparse
import logging
import random
import time

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)"
    )
    argparser.add_argument(
        "-p", "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)"
    )
    argparser.add_argument(
        "-n", "--number-of-vehicles",
        metavar="N",
        default=30,
        type=int,
        help="Number of vehicles (default: 30)",
    )
    argparser.add_argument(
        "-w", "--number-of-walkers",
        metavar="W",
        default=10,
        type=int,
        help="Number of walkers (default: 10)",
    )
    argparser.add_argument(
        "--tm-port",
        metavar="P",
        default=8000,
        type=int,
        help="Port to communicate with traffic manager (default: 8000)",
    )
    argparser.add_argument(
        "-s", "--seed",
        metavar="S",
        type=int,
        help="Set random device seed and deterministic mode for Traffic Manager"
    )

    args = argparser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    random.seed(args.seed if args.seed is not None else int(time.time()))
    
    try:
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)
        traffic_manager.set_synchronous_mode(True)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
        else:
            synchronous_master = False
        world.apply_settings(settings)

        blueprints = world.get_blueprint_library()
        vehicle_blueprints = blueprints.filter("vehicle.*")
        walker_blueprints = blueprints.filter("walker.pedestrian.*")
        
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)

        if args.number_of_vehicles < num_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > num_spawn_points:
            msg = f"Requested {args.number_of_vehicles} vehicles but could only find {num_spawn_points} spawn points"
            logging.warning(msg)
            args.number_of_vehicles = num_spawn_points

        #region Spawn Vehicles
        batch = []
        for n, tranform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(vehicle_blueprints)
            if blueprint.has_attribute("color"):
                color = random.choice(blueprint.get_attribute("color").recommended_values)
                blueprint.set_attribute("color", color)
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
                blueprint.set_attribute("driver_id", driver_id)
            blueprint.set_attribute("role_name", "autopilot")

            batch.append(SpawnActor(blueprint, tranform)
                         .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
        
        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        vehicle_actors = world.get_actors(vehicles_list)
        for actor in vehicle_actors:
            traffic_manager.update_vehicle_lights(actor, True)
            
        logging.info(f"Spawned {len(vehicles_list)} vehicles, press Ctrl+C to exit")

        #endregion

        while True:
            if synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    finally:
        if synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        
        logging.info(f"Destroying {len(vehicles_list)}")
        client.apply_batch([DestroyActor(x) for x in vehicles_list])

        time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        logging.info("Simulator script done")