import viser
import os
import numpy as np
import time
import datetime
import pytz
import torch
import argparse
import threading
import trimesh
import trimesh.creation
import json
import h5py
import matplotlib.cm as cm
import signal
import asyncio
import websockets

from gui.dataloader import CollaborativeDataloader
from gui.model import CollaborativeSegmentationModel
from gui.constants import (
    OBJECT_CLICK_COLOR, 
    BACKGROUND_CLICK_COLOR, 
    PRISM_CLASS_COLORS,
    PRISM_CLASS_IDS,
    FURNITURE_CLASS_COLORS,
    FURNITURE_CLASS_IDS,
    SCANNET_CLASS_IDS,
    SCANNET_CLASS_COLORS
    )
from gui.utils import get_boundary_points, get_extremity_points
from pathlib import Path
from pynput import mouse
import subprocess

class ViserHub:
    """ GUI for Collaborative Segementation using Viser """

    def __init__(self, config):
        self.config = config

        datasets = config.dataset.split(",")
        exps = config.exp_name.split(",")

        print(f"Loading model for dataset(s): {datasets} and experiment(s): {exps}")
        num_needed_models = len(datasets) * len(exps)
        
        self.model_stuff = {}

        self.all_scene_names = []
        for dataset in datasets:
            dataloader = CollaborativeDataloader(dataset_name=dataset)
            

            if dataloader.dataset_type == "ScanNetDataset":
                #"scene0144_00" - This one is a maybe (cause of the picture being weird)
                scene_names = ["scene0690_01", "scene0652_00", "scene0616_00", "scene0314_00", "scene0378_00", "scene0307_00", "scene0435_03", "scene0019_00", "scene0046_01", "scene0050_02", "scene0011_00", "scene0300_00", "scene0328_00", "scene0633_00", "scene0678_01", "scene0139_00", "scene0598_00"]
            elif dataloader.dataset_type == "FurnitureDataset":
                scene_names = ["scene_1527", "scene_1589", "scene_1634", "scene_1698", "scene_1721", "scene_1806", "scene_1912", "scene_1975"]
            elif dataloader.dataset_type == "PrismDataset":
                scene_names = ["scene_47", "scene_93", "scene_128", "scene_196", "scene_274", "scene_319", "scene_386", "scene_442", "scene_497"]
            elif dataloader.dataset_type == "WildScenesDataset":
                scene_names = []
            
            self.model_stuff[dataset] = {
                "dataloader": dataloader}
            # Now we randomly assign an equal amount of scenes to each model
            scene_count = len(scene_names)
            num_models = len(exps)
            scenes_per_model = scene_count // num_models
            remainder = scene_count % num_models

            start = 0
            for i, exp in enumerate(exps):
                extra = 1 if i < remainder else 0
                end = start + scenes_per_model + extra

                assigned_scenes = scene_names[start:end]
                start = end
                
                # model = CollaborativeSegmentationModel(dataset=dataset, exp_name=exp)
                # self.model_stuff[dataset][exp] = {
                #     "model": model,
                #     "assigned_scenes": assigned_scenes,
                # }

                for name in assigned_scenes:
                    self.all_scene_names.append({
                        "dataset": dataset,
                        "exp": exp,
                        "scene_name": name,
                    })

            
        # Now we want to makes sure that the order of scenes is the same - so far we'll do one dataset and then the next, I'm not sure if it's better to do that or not.
        self.scene_iter = iter(self.all_scene_names)
        self.current_dataset = None
        self.current_exp = None
        

        # If no assigned scenes and there is only 1 dataset, then we just go through all it's scenes
        if len(self.all_scene_names) == 0  and len(datasets) == 1 and len(exps) == 1:
            dataloader = self.model_stuff[datasets[0]]["dataloader"]  
            path = dataloader.dataset_path
            self.all_scene_names += [
                {
                    "dataset": dataset,
                    "exp": exp,
                    "scene_name": f
                }
                for f in os.listdir(path)
                if os.path.isdir(os.path.join(path, f))
                ]  
        
        
        # for dataset in datasets:
        #     for exp in exps:
        #         model = CollaborativeSegmentationModel(config)
        #         self.models[f"{dataset}_{exp}"] = model

        # self.model = CollaborativeSegmentationModel(config)
        
        # Point cloud colour storage
        self.base_colors = None
        self.class_colour_map = None
        self.class_labels = None

        self.num_classes = len(self.class_labels) if self.class_labels is not None else 0
        self.shutdown_flag = False
        self.shutdown = False
        

        # Click Management
        # The click is a dictionary of interaction periods, each period contains a list of clicks
        # Each click is a dictionary with keys 'position', 'type' (object/background), 'time'
        self.click_idx = {'0': []}
        self.click_time_idx = {'0': []}
        self.click_positions = {'0': []}
        self.cur_obj_idx = -1
        self.cur_obj_name = None
        self.last_key_pressed_time = round(time.time() * 1000)
        self.num_clicks = 0#{"0": 0}
        self.max_clicks = 30
        self.interactive_periods = 0
        self.radius = 0.05
        

        # Visualisation Considerations
        self.vis_ground_truth = False
        self.current_period = 0
        self.current_vis_period = 0
        self.entropy_vis = False
        self.base_vis = False

        # Now we deal with the viser application stuff
        self.server = viser.ViserServer(host='0.0.0.0', port=8085)
        # self.server.request_share_url()


        # Point cloud handling
        self.point_cloud = self.server.scene.add_point_cloud(
            "point_cloud",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.float32),
            visible=False,
            point_size= 0.01,
        )
        self.current_scene_name = None
        self.scene_start_time = time.time()
        self.last_selected_point = "None"

        # Camera handling - we are going to throttle it to avoid overloading the server with updates
        self.__last_cam_sample = 0.0
        self.CAM_DT = 0.05  # Should be equivalent to 20 Hz
        self.camera_trajectories = []

        # Mouse Handling
        self.mouse_listener = None
        self.last_mouse_position = (None,None)
        self.mouse_positions = []
        self.last_mouse_record = 0.0

        # Gaze handling
        self.gaze_positions = []
        self.last_gaze_record = 0.0
        self.GAZE_DT = 0.05  # Should be equivalent to 20 Hz
        self.gaze_enabled = True

        

        # Now we setup the database for saving session data - becaause we want per scene data everytime we make a new scene we will make a new h5 group
        save_path = Path(f"/workspace/collab3dPerception/gui_sessions/{self.config.user_name}")
        save_path.mkdir(parents=True, exist_ok=True)
        self.session_data = h5py.File(save_path / f"session_data.h5", "w")
        # session_parameters = self.session_data.require_group("Session Parameters")
        aedt_timezone = pytz.timezone('Australia/Sydney')
        self.session_data.attrs["Date/Time"] = datetime.datetime.now(aedt_timezone).strftime("%m/%d/%Y, %H:%M:%S")
        self.session_data.attrs["User ID"] = self.config.user_name
        

        # Screen recording handling
        self.record = True
        self.record_dir = save_path / "scene_recordings"
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.process = None
        


        # Client setup stuff - for tracking the user profile
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            self.client = client

            @self.client.camera.on_update
            def _(camera: viser.CameraHandle) -> None:
                if self.shutdown == True:
                    return
                if self.current_scene_name is not None:
                    current_scene_time = time.time() - self.scene_start_time
                    
                    if current_scene_time - self.__last_cam_sample > self.CAM_DT:
                        self.__last_cam_sample = current_scene_time

                        self.camera_trajectories.append({
                            'time': current_scene_time,
                            'position': camera.position,
                            'rotation': camera.wxyz, 
                        })

        @self.server.on_client_disconnect
        def _(client: viser.ClientHandle) -> None:
            self.client = None
            # We also can do something later here to save the state of the session so it can be effectively resumed?


        # Click Info 
        with self.server.gui.add_folder("Click Information"):
            self.click_info = self.server.gui.add_markdown(
                f"""
                Number of Clicks: {self.num_clicks}/{self.max_clicks}\n
                Current Scene: None Loaded\n
                Current Dataset: None\n
                Current Model: None\n
                """
            )

            self.interaction_allowed = self.server.gui.add_checkbox(
                "Interaction Allowed:",
                initial_value=False,
                disabled=True
            )

            self.object_selector = self.server.gui.add_dropdown(
                "Select Object Class:",
                options=[self.class_labels[label] for label in self.class_labels] if self.class_labels is not None else ["None"],
                disabled=True
            )


            self.colour_preview = self.server.gui.add_html("")#self.server.gui.add_rgb("test", initial_value=(0,0,0), disabled=True)#self.server.gui.add_markdown("")
            self.colour_preview.content = f"""
                                        <div style="
                                            display: flex;
                                            align-items: center;
                                            width: 100%;
                                            font-family: sans-serif;
                                            font-size: 14px;
                                        ">
                                            <div style="
                                                width: 16px;
                                                height: 16px;
                                                min-width: 16px;
                                                background-color: {000000};
                                                border-radius: 3px;
                                                border: 1px solid rgba(0,0,0,0.25);
                                                margin-right: 8px;
                                            "></div>

                                            <div style="
                                                white-space: nowrap;
                                            ">
                                                {self.cur_obj_name}
                                            </div>
                                        </div>
                                        """   
            # self.object_select = self.server.gui.add_button_group(
            #     label="",
            #     options = ("↑", "↓"),
            # )

        # Visalisation Controls
        with self.server.gui.add_folder("Visualization"):
            self.interactive_period = self.server.gui.add_markdown(
                f"""
                **Interactive Period:** {self.current_period}\n
                **Current Visualized Period:** {self.current_vis_period}\n
                """
            )

            # self.visualise_options = self.server.gui.add_button(
            #     "Cycle Interactions",
            # )
            self.prediction_selector = self.server.gui.add_dropdown(
                label="Prediction Visualisation",
                options = ["None"],
                disabled=True
            )

            self.additional_vis_options = self.server.gui.add_button_group(
                label="Additional Visualisation Options",
                options = ("Entropy", "Base Colour"),
            )





        # Control Buttons
        with self.server.gui.add_folder("Controls"):

            self.scene_nav = self.server.gui.add_button_group(
                label="Scene Navigation",
                options = ("Previous", "Next"),
            )

            self.run_seg_button = self.server.gui.add_button(
                "Run / Save [Enter]",
            )

            self.save_quit_button = self.server.gui.add_button(
                "Save and Quit [Esc]",
            )

        

        with self.server.gui.add_modal("Commencement Confirmation") as modal:
            self.server.gui.add_markdown(
                "Are you ready to start?"
            )
            button = self.server.gui.add_button(
                "Start",
                
            ).on_click(lambda _: (self.commence(),modal.close()))

        

        # Callback Definitions
        @self.run_seg_button.on_click
        def _(_) -> None:
            self.__run_segmentation()
            
        @self.scene_nav.on_click
        def _(_) -> None:
            self.__on_scene_nav(self.scene_nav.value)
        
        @self.save_quit_button.on_click
        def _(_) -> None:
            # print("Shutting down Viser GUI")
            if self.shutdown == True:
                return

            if self.mouse_listener is not None:
                self.mouse_listener.stop()
            self.stop_recording()

            
            self.shutdown = True
            self.save_scene_data()
            self.session_data.close()
            time.sleep(0.5)
            self.server.stop()
            self.shutdown_flag = True
        
            

        # @self.visualise_options.on_click
        # def _(_) -> None:
        #     self.__cycle_interactions()

        @self.prediction_selector.on_update
        def _(_) -> None:
            if self.prediction_selector.value is None or self.prediction_selector.value == "None":
                return
            elif self.prediction_selector.value == "Ground Truth":
                prediction_colors = np.array([self.class_colour_map[label] for label in self.ground_truth])
            else:
                self.current_vis_period = int(self.prediction_selector.value.strip(" ")[-1])
                prediction_colors = np.array([self.class_colour_map[label] for label in self.predictions[self.current_vis_period]])
            
            self.point_cloud.colors = prediction_colors
            self.base_vis = False
            self.entropy_vis = False
            self.__update_scene_text()

        @self.interaction_allowed.on_update
        def _(_) -> None:
            if self.interaction_allowed.value:
                # print("Interaction mode enabled.")
                @self.server.on_pointer_event(event_type='click')
                def _(event: viser.ScenePointerEvent) -> None:
                    self.__on_click(event)


            else:
                # print("Interaction mode disabled.")
                self.server.remove_pointer_callback()

        @self.additional_vis_options.on_click
        def _(_) -> None:
            if self.additional_vis_options.value == "Entropy":
                if getattr(self, "entropies", None) is not None and len(self.entropies) > 0:
                    self.__entropy_vis()
                else:
                    self.notify_client = self.client.add_notification(title="Visualisation Error", body="No predictions available to visualize entropy.", auto_close_seconds=5.0)
            elif self.additional_vis_options.value == "Base Colour":
                if getattr(self, "base_colors", None ) is not None:
                    self.__base_color_vis()

        @self.object_selector.on_update
        def _(_) -> None:
            selected_label = self.object_selector.value
            if selected_label is not None and selected_label != "None":
                self.cur_obj_name = selected_label
                # Find the key in class_labels dict that has the selected_label as value
                self.cur_obj_idx = [key for key, value in self.class_labels.items() if value == selected_label][0]
                self.client.add_notification(title="Object Selection", body=f"Selected object class: {self.cur_obj_name} (Index: {self.cur_obj_idx})", auto_close_seconds=5.0)
                
                colour = self.class_colour_map[self.cur_obj_idx]
                hex_colour = rgb_to_hex(colour)

                self.colour_preview.content = f"""
                                        <div style="
                                            display: flex;
                                            justify-content: center;
                                            align-items: center;
                                            gap: 8px;
                                            font-family: sans-serif;
                                            font-size: 14px;
                                        ">
                                            <div style="
                                                width: 16px;
                                                height: 16px;
                                                background-color: {hex_colour};
                                                border-radius: 3px;
                                                border: 1px solid rgba(0,0,0,0.25);
                                            "></div>
                                            <div style="font-weight: 500;">
                                                {selected_label}
                                            </div>
                                        </div>
                                        """            
            else:
                self.cur_obj_name = None
                self.cur_obj_idx = -1
                self.client.add_notification(title="Object Selection", body="No object class selected.", auto_close_seconds=5.0)


        # @self.server.scene.on_gaze_event()
        # def _(event: viser.SceneGazeEvent) -> None:
        #     print("Is this thing on?")
        #     self.__on_gaze(event)



    # def __on_gaze(self, event: viser.SceneGazeEvent) -> None:
    #     print("test")
    #     if not self.gaze_enabled:
    #         return
    #     if self.current_scene_name is None:
    #         return
    #     print("hi")
    #     t = time.time() - self.scene_start_time
    #     if t - self.last_gaze_record < self.GAZE_DT:
    #         return
        
    #     self.last_gaze_record = t

    #     # event.screen_pos is (x, y) normalised [0,1] in opencv coords
    #     x_norm, y_norm = event.screen_pos
    #     self.gaze_positions.append((x_norm, y_norm, t))
        

    # Functions
    def commence(self):
        # Find the active window
        wid = subprocess.check_output(
            ["xdotool", "getactivewindow"]
        ).decode().strip()

        # Get geometry in shell format
        geom_output = subprocess.check_output(
            ["xdotool", "getwindowgeometry", "--shell", wid]
        ).decode()

        self.geom = {}
        for line in geom_output.splitlines():
            if "=" in line:
                k, v = line.split("=")
                self.geom[k] = int(v)

    def start_recording(self, fps=30):
        """
            Start the screen recording using the active window.
        """

        timestamp = int(self.scene_start_time)

        output_path = self.record_dir / f"{self.current_scene_name}_{timestamp}.mp4"


        x = self.geom["X"]
        y = self.geom["Y"]
        width = self.geom["WIDTH"]
        height = self.geom["HEIGHT"]

        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1

        if self.geom is None:
            self.client.add_notification("Screen Recording", body="Screen recording did not start correctly", with_close_button=True)
            self.process = None
            return
    
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "x11grab",
            "-framerate", str(fps),
            "-video_size", f"{width}x{height}",
            "-i", f"{os.environ.get('DISPLAY', ':1')}+{x},{y}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_recording(self):
        if self.process:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        
    def mouse_movement(self, x, y):
        win_x = self.geom["X"]
        win_y = self.geom["Y"]
        win_w = self.geom["WIDTH"]
        win_h = self.geom["HEIGHT"]
        timestamp = time.time() - self.scene_start_time

        if timestamp - self.last_mouse_record > self.CAM_DT:
            self.last_mouse_record = timestamp
            if win_x <= x <= win_x + win_w and win_y <= y <= win_y + win_h:
                x_norm = (x - win_x) / win_w
                y_norm = (y - win_y) / win_h
                self.last_mouse_position = (x_norm, y_norm)
                self.mouse_positions.append((x_norm, y_norm, timestamp))
            else:
                self.last_mouse_position = (None, None)
                self.mouse_positions.append((None, None, timestamp))


    def __run_segmentation(self):
        if self.point_cloud is None or self.current_scene_name is None:

            self.client.add_notification(title="Run Error", body="Need to load a point cloud first", auto_close_seconds=5.0)
            return
        else:
            self.__model_predict()
            self.__update_scene_text()

    def __on_scene_nav(self, selection: str):
        if selection == "Previous":
            # print("Previously on Total Drama Island")
            self.__prev_scene()

        elif selection == "Next":
            # print("Next time on Total Drama Island")
            self.__next_scene()
            
    
    def __update_scene_text(self):

        self.click_info.content = f"""
            Number of Clicks: {self.num_clicks}/{self.max_clicks}\n
            Current Scene: {self.current_scene_name}\n
            Current Dataset: {self.current_dataset}\n
            Current Model: {self.current_exp}\n
            """
        
        if self.current_vis_period == -1:
            vis_period_text = "Ground Truth"
        else:
            vis_period_text = str(self.current_vis_period)

        self.interactive_period.content = f"""
            **Interactive Period:** {self.current_period}\n
            **Visualized Period:** {vis_period_text}\n
            """
        # if self.current_dataset == "scannet":
        #     self.object_selector.options = [self.class_labels[VALID_CLASS_IDS_20[label]] for label in self.class_labels] if self.class_labels is not None else ["None"]
        # else:
        # self.object_selector.options = [self.class_labels[label] for label in self.class_labels] if self.class_labels is not None else ["None"]
        
    # def __cycle_interactions(self):
            
    #     if hasattr(self, "predictions"):
    #         if self.current_vis_period + 1 < len(self.predictions):
    #             self.current_vis_period += 1
    #             prediction_colors = np.array([self.class_colour_map[label] for label in self.predictions[self.current_vis_period]])

    #         else:
    #             self.current_vis_period = -1
    #             if self.ground_truth is not None:
    #                 prediction_colors = np.array([self.class_colour_map[label] for label in self.ground_truth])
    #             else:
    #                 self.client.add_notification(title="Visualisation Error", body="No ground truth available for this scene.", auto_close_seconds=5.0)
    #                 self.current_vis_period = 0
    #                 prediction_colors = np.array([self.class_colour_map[label] for label in self.predictions[self.current_vis_period]])

    #         self.point_cloud.colors = prediction_colors

    #         self.__update_scene_text()

    def __prev_scene(self):
        pass

    def __next_scene(self):
        """
        Loads the next scene through the data loader
        
        :param self: Description
        """
        if self.mouse_listener is not None:
            self.mouse_listener.stop()
        self.stop_recording()

        print(self.gaze_positions)
        self.gaze_positions = []
        self.last_gaze_record = 0.0

        # First clear all meshes related to clicks
        for period in self.click_idx:
            for click in self.click_idx[period]:
                if 'handle' in click:
                    click['handle'].remove()
        if self.current_scene_name is not None:
            self.save_scene_data()


        # Enable all GUI elements
        self.interaction_allowed.disabled = False
        self.object_selector.disabled = False
        self.additional_vis_options.disabled = False


        # Now we load the next scene through the iterator
        next_scene_info = next(self.scene_iter, None)
        if next_scene_info is None:
            self.client.add_notification(title="End of Scenes", body="No more scenes available.", auto_close_seconds=5.0)
            print("nothing left to load")
            return
        
        # print(next_scene_info)
        dataset = next_scene_info["dataset"]
        exp = next_scene_info["exp"]
        scene_name = next_scene_info["scene_name"]
        
        # print(self.model_stuff[dataset])
        self.dataloader = self.model_stuff[dataset]["dataloader"]
        # self.model = self.model_stuff[dataset][exp]["model"]
        self.model = self.__get_model_for_scene(dataset, exp, scene_name)

        
        self.current_scene_name = scene_name
        self.client.add_notification(title="Scene Loaded", body=f"Loaded scene: {scene_name} from dataset: {dataset} using model: {exp}", auto_close_seconds=5.0)

        if self.dataloader.dataset_type == "ScanNetDataset":
            self.class_colour_map = SCANNET_CLASS_COLORS
            self.class_labels = SCANNET_CLASS_IDS
        elif self.dataloader.dataset_type == "FurnitureDataset":
            self.class_colour_map = FURNITURE_CLASS_COLORS
            self.class_labels = FURNITURE_CLASS_IDS
        elif self.dataloader.dataset_type == "PrismDataset":
            self.class_colour_map = PRISM_CLASS_COLORS
            self.class_labels = PRISM_CLASS_IDS
        # elif self.dataloader.dataset_type == "WildScenesDataset":
        #     self.class_colour_map = WILDSCENES_CLASS_COLORS
        #     self.class_labels = WILDSCENES_CLASS_IDS
        else:
            self.class_colour_map = None
            self.class_labels = None

        self.num_classes = len(self.class_labels) if self.class_labels is not None else 0
        self.current_dataset = dataset
        self.current_exp = exp
        self.object_selector.options=[self.class_labels[label] for label in self.class_labels] if self.class_labels is not None else ["None"]


        # Now we actually load the data
        self.points, self.base_colors, self.ground_truth, self.current_scene_data = self.dataloader.load_scene_by_name(scene_name=self.current_scene_name)
        self.ignore_mask = self.ground_truth == -1
        self.prediction_selector.options = ["Ground Truth"]

        self.predictions = []
        self.entropies = []
        self.corrections = []
        self.interactions = []

        self.scene_start_time = time.time()
        self.__model_predict()

        self.num_clicks = 0
        self.__update_scene_text()

        self.click_idx = {'0': []}
        self.camera_trajectories = []
        self.__last_cam_sample = 0

        if self.record:
            self.start_recording()
        self.mouse_listener = mouse.Listener(on_move = self.mouse_movement)
        self.last_mouse_position = (None,None)
        self.mouse_positions = []
        self.mouse_listener.start()



        

    def __get_model_for_scene(self, dataset, exp, scene_name):
        """
        Docstring for __get_model_for_scene
        
        :param self: Description
        :param dataset: Description
        :param exp: Description
        :param scene_name: Description
        """

        model = CollaborativeSegmentationModel(dataset=dataset, exp_name=exp)

        return model    

    def __model_predict(self):
        """
        Docstring for __model_predict
        
        :param self: Description
        """
        
        output_dict  = self.model.predict(click_idx=self.click_idx, scene_name=self.current_scene_name, scene_data=self.current_scene_data)
        prediction = output_dict["prediction"]        
        self.prediction_selector.disabled = False
        # If the ground truth is -1 at a point change the prediciton to be -1 at that point.
        # for idx, val in enumerate(prediction):
        #     if self.ground_truth[idx] == -1:
        #         prediction[idx] = -1
        #         # print("fixing predictions")
        prediction[self.ignore_mask] = -1
        # print()
        if len(self.predictions) == 0:
            self.current_period = 0
            self.current_vis_period = 0

        else:
            self.current_period += 1
            self.current_vis_period +=1

            # Check the difference between this prediction and the last one
            if np.array_equal(prediction, self.predictions[-1]):
                print("Did it actually work? No change in prediction after interaction.")

        self.predictions.append(prediction)
        self.entropies.append(output_dict["entropy"])
        self.corrections.append(output_dict["corrections"])
        self.interactions.append(output_dict["interactions"])

        self.click_idx[str(self.current_period)] = []

        # print(f"Unique prediction colour values: {np.unique(prediction_colors, axis=0)}")
        # Visualise the prediction
        self.point_cloud.points = self.points
        # self.point_cloud.colors = prediction_colors
        self.point_cloud.visible = True

        options = ["Ground Truth"]
        for period in range(0, self.current_period+1):
            options += [f"Prediction {period}"]
        
        self.prediction_selector.options = options
        self.prediction_selector.value = options[-1]


    def __entropy_vis(self):
        """
            When triggered will alternate the view to show the entropy map instead of the predicted labels
        """
        if self.entropy_vis == False:
            self.entropy_vis = True
            self.base_vis = False
            # Get the entropy values for the current prediction
            current_entropy = self.entropies[self.current_vis_period]

            # Normalize the entropy values to [0, 1]
            norm_entropy = current_entropy / np.log(self.num_classes)
            norm_entropy = np.clip(norm_entropy, 0, 1)

            # Make it so that any points that are not meant to be used (the -1) class are black
            norm_entropy[self.ignore_mask] = 0

            # Use a colormap to convert normalized entropy to colors
            colormap = cm.get_cmap('inferno')
            entropy_colors = colormap(norm_entropy)[:, :3]  # Get RGB values, ignore alpha

            
            
            self.point_cloud.colors = (entropy_colors * 255).astype(np.uint8)
            self.prediction_selector.value = None


            with self.server.gui.add_modal("Entropy Heatmap") as modal:
                height = 40
                width = 256  # thickness in pixels
                gradient = np.linspace(0, 1, width)[None, :]  # (1, W)
                gradient = np.repeat(gradient, height, axis=0)  # (H, W)            colors = cm.inferno(gradient)[:, :3]
                colors = cm.inferno(gradient)[:, :, :3]  # (H, W, 3), RGB
                legend_img = (colors * 255).astype(np.uint8)[None, :, :]  # (1, 256, 3)
                entropy_gradient = self.server.gui.add_image(image=legend_img, label="\tHigh Confidence -> Low Confidence")

                exit_button = self.server.gui.add_button(
                    label="Exit"
                ).on_click(lambda _: modal.close())

               


        else:
            # Revert to the original prediction colors
            prediction = self.predictions[self.current_vis_period]
            prediction_colors = np.array([self.class_colour_map[label] for label in prediction])
            self.point_cloud.colors = prediction_colors

            self.entropy_vis = False


    def __base_color_vis(self):
        """
        When triggered will alternate the view to show the base colour information
        """
        if self.base_vis == False:
            self.base_vis = True
            self.entropy_vis = False
        

            self.point_cloud.colors = (self.base_colors * 255).astype(np.uint8)
            self.prediction_selector.value = None
        else:
            # Revert to the original prediction colors
            prediction = self.predictions[self.current_vis_period]
            prediction_colors = np.array([self.class_colour_map[label] for label in prediction])
            self.point_cloud.colors = prediction_colors

            self.base_vis = False


    def __on_click(self, event: viser.ScenePointerEvent):
        # Hard gate 1: interaction mode
        if not self.interaction_allowed.value:
            return
        
        if self.shutdown == True:
            return


        if self.points is None:
            self.client.add_notification(title="Click error", body="No point cloud loaded.", auto_close_seconds=5.0)
            return
        
        if self.num_clicks >= self.max_clicks:
            self.client.add_notification(title="Click error", body="Run out of clicks for this scene")
            self.interaction_allowed.value = False
            return

        # print("Ray Origin",event.ray_origin)
        # print("Ray Direction",event.ray_direction)

        o = np.asarray(event.ray_origin, dtype=np.float32).reshape(1, 3)
        d = np.asarray(event.ray_direction, dtype=np.float32) / np.linalg.norm(np.asarray(event.ray_direction, dtype=np.float32))

        # Vector from ray origin to points
        v = self.points - o

        # Project onto ray
        t = np.dot(v, d)

        # print(t)
        epsilon = 0.01  # Selection radius
        # Only consider points in front of camera
        valid = t > 0
        if not np.any(valid):
            return None, None

        v = v[valid]
        t = t[valid]
        pts = self.points[valid]

        # Perpendicular distance to ray
        perp = v - t[:, None] * d
        dist = np.linalg.norm(perp, axis=1)

        # Within selection radius
        hit_mask = dist < epsilon
        if not np.any(hit_mask):
            self.client.add_notification(title="Click Error", body="No hits within selection radius.", auto_close_seconds=5.0)
            return None, None

        # First hit along ray
        idx = np.argmin(t[hit_mask])
        hit_point = pts[hit_mask][idx]
        hit_t = t[hit_mask][idx]


        # We need to check if the ground truth label of the click is -1 (if so send a notifcation that point wasn't registered cause of background clicks)
        gt_mask = self.ground_truth[valid][hit_mask]
        if gt_mask[idx] == -1:
            self.client.add_notification(title="Click Error", body="Not a valid class to interact with", auto_close_seconds=5.0)
            return None, None
            

        # print(f"Hit point at {hit_point} (t={hit_t})") 
        click_handle = self.__visualize_click(hit_point)


        self.last_selected_point = hit_point.tolist()
        self.num_clicks += 1

        self.click_idx[str(self.current_period)].append(
            {
                'position': hit_point,
                "class": self.cur_obj_idx,
                'time': time.time() - self.scene_start_time,
                'handle': click_handle,
            }
        )

        print(f"Registered click at {hit_point} for class {self.cur_obj_name} (Index: {self.cur_obj_idx})")

        self.__update_scene_text()
        self.interaction_allowed.value = False
        # print(
        #     f"Ctrl+Click registered | idx={point_idx} | dist={distance:.4f} | period={self.current_period}"
        # )

        # self.__update_scene_text()
        # self.__visualize_click(selected_point)


    def __visualize_click(self, position: np.ndarray):
        sphere = trimesh.creation.icosphere(radius=self.radius)
        sphere.apply_translation(position)

        color = (
            self.class_colour_map[self.cur_obj_idx]
            if self.cur_obj_idx is not None
            else BACKGROUND_CLICK_COLOR
        )

        color = np.asarray(color, dtype=np.float32)

        # If colors are 0–255 scale
        if color.max() > 1.0:
            dark_color = np.clip(color * 0.6, 0, 255).astype(np.uint8)
        else:
            # If colors are 0–1 scale
            dark_color = np.clip(color * 0.6, 0, 1.0)

        sphere.visual.face_colors = np.tile(
            dark_color,
            (sphere.faces.shape[0], 1),
        )

        mesh = self.server.scene.add_mesh_trimesh(
            name=f"click_{self.current_period}_{self.num_clicks}",
            mesh=sphere,
        )

        

        @mesh.on_click
        def _(_) -> None:
            # On click create a popup notification to confirm deletion
            with self.server.gui.add_modal("Confirm Deletion") as modal:
                self.server.gui.add_markdown(
                    f"Are you sure you want to delete click at {position}?"
                )
                delete_button = self.server.gui.add_button_group(
                    label="",
                    options= ("Yes", "No"))

                @delete_button.on_click
                def _(_) -> None:
                    if delete_button.value == "Yes":

                        for idx,click in enumerate(self.click_idx[str(self.current_period)]):
                            if click.get('handle') == mesh:
                                self.click_idx[str(self.current_period)].pop(idx)
                                mesh.remove()
                                self.num_clicks -= 1
                                self.last_selected_point = "Deleted"
                                self.__update_scene_text()


                                break
                    modal.close()

        return mesh
    

    def __reset_camera_pos(self):
        if self.client is not None:
            self.client.camera.position = np.asarray((-1.5, 5, 4), dtype=np.float64)
            self.client.camera.wxyz = np.asarray((0, 0, 1, -0.5), dtype=np.float64)



# ------ Save Functions ------


    def save_scene_data(self):
        """
        Saves the scene data to a specified path.
        
        :param self: Description
        """
        if self.current_scene_name == None:
            return
        # We're going to save it as a h5 database - as it allows for better structure and we can handle a scene by scene basis
        save_file = self.session_data
        scene_grp = save_file.require_group(self.current_dataset).require_group(self.current_scene_name)
        scene_grp.attrs['num_clicks'] = self.num_clicks
        scene_grp.attrs['num_periods'] = self.current_period 
        scene_grp.attrs['session_time'] = time.time() - self.scene_start_time
        scene_grp.attrs['dataset'] = self.current_dataset 
        scene_grp.attrs['model_type'] = self.current_exp
        scene_grp.attrs["scene_name"] = self.current_scene_name



            # ---- clicks ----
        clicks = scene_grp.require_group("clicks")

        for click_period in self.click_idx.keys():
            click_grp = clicks.require_group(f"Period {click_period}")
            
            if len(self.click_idx[click_period]) == 0:
                click_grp.attrs["Interaction"] = False
            else:
                click_grp.attrs["Interaction"] = True
                click_grp.create_dataset(
                    "time",
                    data=np.array([c["time"] for c in self.click_idx[click_period]]),
                )
                click_grp.create_dataset(
                    "position",
                    data=np.stack([c["position"] for c in self.click_idx[click_period]]),
                )
                click_grp.create_dataset(
                    "class",
                    data=np.array([c["class"] for c in self.click_idx[click_period]]),
                )

        # ---- camera trajectories ----
        cam = scene_grp.require_group("camera")
        if len(self.camera_trajectories) == 0:
            cam.attrs["num_samples"] = 0
            
        else:
            cam.attrs["num_samples"] = len(self.camera_trajectories)
            cam.create_dataset(
                "time",
                data=np.array([c["time"] for c in self.camera_trajectories]),
            )
            cam.create_dataset(
                "position",
                data=np.stack([c["position"] for c in self.camera_trajectories]),
            )
            cam.create_dataset(
                "rotation",
                data=np.stack([c["rotation"] for c in self.camera_trajectories]),
            )

        # ---- Model based parameters -----
        # Stack the self arrays
        pred = np.stack(self.predictions, axis=1)
        entropies = np.stack(self.entropies, axis=1)
        corrections = np.stack(self.corrections, axis=2)
        interactions = np.stack(self.interactions, axis=1)

        mouse_data = np.array(self.mouse_positions, dtype=np.float32)
        scene_grp.create_dataset("mouse", data=mouse_data)

        # We also need to save the same things that test saves so the visualisation engine will work the same way
        model = scene_grp.require_group("Model")
        model.create_dataset(
            "predictions", data=pred, dtype=pred.dtype,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

        model.create_dataset(
            "entropies", data=entropies, dtype=entropies.dtype,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

        model.create_dataset(
            "corrections", data=corrections, dtype=corrections.dtype,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

        model.create_dataset(
            "interactions", data=interactions, dtype=interactions.dtype,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

        model.attrs["remaining_budget"] = self.max_clicks - self.num_clicks




def rgb_to_hex(color):
    color = np.asarray(color)

    # If color is float in [0,1], scale to [0,255]
    if color.max() <= 1.0:
        color = (color * 255.0)

    color = color.astype(int)

    return "#{:02x}{:02x}{:02x}".format(*color[:3])

def color_square(color):
    return "■"
