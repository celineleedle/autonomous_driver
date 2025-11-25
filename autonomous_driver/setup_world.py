#1/usr/bin/env python

"""Script to setup parameters and generate world for data gathering"""

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
        "-m", "--map",
        metavar="M",
        help="Name of map to load in world"
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
        "--seedt",
        metavar="T",
        type=int,
        help="Set random device seed and deterministic mode for Traffic Manager"
    )
    argparser.add_argument(
        "--seedw",
        metavar="W",
        type=int,
        help="Set the seed for pedestrians module",
    )
    argparser.add_argument(
        "--running",
        metavar="R",
        default=0.31,
        type=float,
        help="Percentage of pedestrians that will run",
    )
    argparser.add_argument(
        "--crossing",
        metavar="C",
        default=0.47,
        type=float,
        help="Percentage of pedestrians that will walk on the road or cross at any point on the road",
    )

    args = argparser.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

    vehicles_list = []
    walkers_list = []
    all_ids = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    random.seed(args.seedt if args.seedt is not None else int(time.time()))
    
    try:
        world = client.get_world()

        if args.map:
            if args.map not in client.get_available_maps():
                raise ValueError("Could not find any map with the given name")
            world = client.load_world(args.map)

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)
        traffic_manager.set_synchronous_mode(True)
        if args.seedt is not None:
            traffic_manager.set_random_device_seed(args.seedt)

        settings = world.get_settings()
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
        else:
            synchronous_master = False
        world.apply_settings(settings)
        world.set_pedestrians_cross_factor(args.crossing)

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


        #region Spawn Walkers
        if args.seedw:
            world.set_pedestrians_seed(args.seedw)
            random.seed(args.seedw)

        spawn_points = []
        for _ in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        
        # spawn walker objects
        batch = []
        walker_speeds = []
        for spawn_point in spawn_points:
            walker_blueprint = random.choice(walker_blueprints)
            if walker_blueprint.has_attribute("is_invincible"):
                walker_blueprint.set_attribute("is_invincible", "false")
            if walker_blueprint.has_attribute("can_use_wheelchair") and random.randint(0, 100) <= 11:
                walker_blueprint.set_attribute("use_wheelchair", "true")
                
            if walker_blueprint.has_attribute("speed"):
                if random.random() > args.running:
                    # walking
                    walker_speeds.append(walker_blueprint.get_attribute("speed").recommended_values[1])
                else:
                    # running
                    walker_speeds.append(walker_blueprint.get_attribute("speed").recommended_values[2])
            else:
                walker_speeds.append(0.0)
            
            batch.append(SpawnActor(walker_blueprint, spawn_point))

        successful_walker_speeds = []
        responses = client.apply_batch_sync(batch, synchronous_master)
        for i in range(len(responses)):
            if responses[i].error:
                logging.error(responses[i].error)
            else:
                walkers_list.append({"id": responses[i].actor_id})
                successful_walker_speeds.append(walker_speeds[i])
        walker_speeds = successful_walker_speeds

        # spawn walker controllers
        batch = []
        walker_controller_blueprint = world.get_blueprint_library().find("controller.ai.walker")
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_blueprint, carla.Transform(), walkers_list[i]["id"]))
        responses = client.apply_batch_sync(batch, synchronous_master)
        for i in range(len(responses)):
            if responses[i].error:
                logging.error(responses[i].error)
            else:
                walkers_list[i]["con"] = responses[i].actor_id
        
        for i in range(len(walkers_list)):
            # controller id first because step by 2 starting at 0 for starting in loop below
            all_ids.append(walkers_list[i]["con"])
            all_ids.append(walkers_list[i]["id"])

        if not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        all_actors = world.get_actors(all_ids)
        for i in range(0, len(all_ids), 2):
            all_actors[i].start()
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            all_actors[i].set_max_speed(float(walker_speeds[int(i/2)]))

        logging.info(f"Spawned {len(walkers_list)} walkers, press Ctrl+C to exit")

        #endregion

        traffic_manager.global_percentage_speed_difference(-7.0)

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
        
        logging.info(f"Destroying {len(vehicles_list)} vehicles")
        client.apply_batch([DestroyActor(x) for x in vehicles_list])

        for i in range(0, len(all_ids), 2):
            all_actors[i].stop()

        logging.info(f"Destroying {len(walkers_list)} walkers")
        client.apply_batch([DestroyActor(x) for x in all_ids])

        time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        logging.info("Simulator script done")