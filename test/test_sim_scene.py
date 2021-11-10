from active_grasp.simulation import Simulation


def main():
    gui = True
    scene_id = "random"
    vgn_path = "../vgn/assets/models/vgn_conv.pth"
    sim = Simulation(gui, scene_id, vgn_path)
    while True:
        sim.reset()


if __name__ == "__main__":
    main()
