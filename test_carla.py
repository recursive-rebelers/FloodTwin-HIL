import carla

def main():
    try:
        print("Connecting to CARLA simulator...")
        # This connects to the simulator running on your machine
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)

        # Retrieve the current world map
        world = client.get_world()
        print(f"Success! Connected to CARLA 0.9.16. Current Map: {world.get_map().name}")

    except Exception as e:
        print(f"Failed to connect. Error: {e}")

if __name__ == '__main__':
    main()