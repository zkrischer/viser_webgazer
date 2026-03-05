"""
Code to start running the GUI

Author: Ze'ev Krischer - Australian Centre for Robotics, Data61
<zeev.krischer@sydney.edu.au>
"""

import argparse
import torch
import time
import threading
import asyncio
import websockets
import json
import webbrowser

from pointcept.models.sparse_unet import interactive_spconv_unet
from gui.mainHub import CollaborativeSegmentationGUI
from gui.viser_hub import ViserHub


def main(_):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    # Set up dataloader and model
    if config.gui_type == "Open3D":
        gui = CollaborativeSegmentationGUI(config)
        # Run the segmentation process - I think this should be part of an overall GUI class rather than the model like AGILE does it
        gui.app.run()
    elif config.gui_type == "Viser":
        gui = ViserHub(config)
        while not gui.shutdown_flag: 
            time.sleep(1)

    else:
        print(f"GUI Type {config.gui_type} is not supported.\n Supported types are Open3D and Viser")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Setup arguments
    parser.add_argument("--user_name", type=str, default="user_00", help="Name of the user, used to select the profile to attach to the model")
    parser.add_argument("--exp_name", type=str, default="nonInteractive", help="The experiment from which to load the pretraining weights.")
    parser.add_argument("--dataset", type=str, default="furniture", help="The dataset to get the test scenes from")
    parser.add_argument("--buffer_size", type=int, default=5, help="The size of the buffer for storing interactions")
    parser.add_argument("--gui_type", type=str, help="The framework used for the gui")
    config = parser.parse_args()
    

    main(config)



