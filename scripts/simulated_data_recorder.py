import os
import threading
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pynput import keyboard

IMG_X = 180  # Camera observation width
IMG_Y = 320  # Camera observation height


class Recorder:
    """Class to simulate and record states, actions, and observations of the Sawyer Robot's end effector"""

    def __init__(self, file: h5py.File, sample_interval: float, store_observations: bool = False):
        """Initialize the recorder with the HDF5 file, sampling interval, and demonstration recording parameters"""
        self.file = file
        self.filename = self.file.filename[0:self.file.filename.index(".")]
        self.sample_interval = sample_interval
        self.store_observations = store_observations
       

        # Custom datatype for storing intera_interface.Limb().endpoint_pose() in a HDF5 dataset
        self.dt = np.dtype([
            ('position', np.float32, (3,)),
            ('orientation', np.float32, (4,))
        ])

        # Demonstration recording initialization
        self.demo_num = 0
        self.recording = False
        self.sample_count = 0
        self.demo_group = None
        self.prev_state = None
        self.record_thread = None

    def setup_group(self, demo_num: int, description: str):
        """Create an HDF5 group for storing timestamps, states, actions, and observations for a given demonstration."""
        demo_group = self.file.create_group(f"demo_{demo_num}")
        demo_group.attrs["description"] = description  # Optional, text-based description of demonstration
        demo_group.attrs["num_samples"] = 0

        # Datasets start with 0 elements but will be resized when storing data
        demo_group.create_dataset("color_images", (0,IMG_X, IMG_Y, 3), maxshape=(None, IMG_X, IMG_Y, 3), dtype="uint8")
        demo_group.create_dataset("depth_images", (0, IMG_X, IMG_Y), maxshape=(None, IMG_X, IMG_Y), dtype="uint8")
        demo_group.create_dataset("actions", (0,), maxshape=(None,), dtype=self.dt)
        
        self.demo_group = demo_group

    def start_recording(self):
        """Begin recording data by initializing a thread to collect samples every sample_interval seconds."""
        self.recording = True
        description = f"Sample demonstration {self.demo_num}"
        # Uncomment the line below to add descriptions to demonstrations
        # description = input("Enter the description for this demonstration...\n")
        self.setup_group(self.demo_num, description)
        self.demo_group.attrs["start_time"] = time.time()
        self.sample_count = 0
        self.prev_state = None
        self.record_thread = threading.Thread(target=self.record_sample_thread, args=())
        print(f"Recording demonstration {self.demo_num}. Press <q> to end the recording.")
        self.record_thread.start()

    def stop_recording(self):
        """Stop the recording process and finalize the data in the current demonstration."""
        if self.recording:
            self.demo_group.attrs["end_time"] = time.time()
            self.demo_group.attrs["num_samples"] = self.sample_count
            self.demo_num += 1
            self.recording = False
            self.record_thread.join()
            print(f"\nDemonstration {self.demo_num} recorded.")
            print("Press <ENTER> to start recording another demonstration or press <q> to exit the program.")

    def record_sample(self):
        """Store endpoint state, action taken since last sample, and an observation within its respective dataset."""
        if self.recording:
            timestamp_time = time.time() - self.demo_group.attrs["start_time"]
            position = np.random.rand(3)  # Random position
            orientation = np.random.rand(4)  # Random orientation

            
            color_images = self.demo_group["color_images"]
            depth_images = self.demo_group["depth_images"]
            actions = self.demo_group["actions"]

           
            color_images.resize((self.sample_count + 1, IMG_X, IMG_Y, 3))
            depth_images.resize((self.sample_count + 1, IMG_X, IMG_Y))
            actions.resize((self.sample_count + 1,))

           
            if self.prev_state is None:  # The first action should just be the first state
                actions[self.sample_count] = (position, orientation)
            else:
                delta_position = position - self.prev_state["position"]

                curr_w, curr_x, curr_y, curr_z = orientation
                prev_w, prev_x, prev_y, prev_z = self.prev_state["orientation"]

                # Compute the conjugate of the previous orientation to get difference in quaternions
                prev_conj_w = prev_w
                prev_conj_x = -prev_x
                prev_conj_y = -prev_y
                prev_conj_z = -prev_z

                # Quaternion multiplication (prev_conjugate * current_orientation)
                delta_w = prev_conj_w * curr_w - prev_conj_x * curr_x - prev_conj_y * curr_y - prev_conj_z * curr_z
                delta_x = prev_conj_w * curr_x + prev_conj_x * curr_w + prev_conj_y * curr_z - prev_conj_z * curr_y
                delta_y = prev_conj_w * curr_y - prev_conj_x * curr_z + prev_conj_y * curr_w + prev_conj_z * curr_x
                delta_z = prev_conj_w * curr_z + prev_conj_x * curr_y - prev_conj_y * curr_x + prev_conj_z * curr_w

                delta_orientation = (delta_x, delta_y, delta_z, delta_w)
                actions[self.sample_count] = (delta_position, delta_orientation)
            self.prev_state = {"position": position, "orientation": orientation}
            
            color_image = np.random.randint(0, 256, (IMG_X, IMG_Y, 3), dtype='uint8')  # Random color image
            color_images[self.sample_count] = color_image

            depth_image = np.random.randint(0, 256, (IMG_X, IMG_Y), dtype='uint8')  # Random depth image
            depth_images[self.sample_count] = depth_image

            if self.store_observations:
                if not os.path.exists(f"dummy/{self.filename}/demo_{self.demo_num}"):
                    os.makedirs(f"dummy/{self.filename}/demo_{self.demo_num}/", exist_ok = True)
                try:
                    plt.imsave(f"dummy/{self.filename}/demo_{self.demo_num}/sample_{self.sample_count}_color.png", color_image)
                    plt.imsave(f"dummy/{self.filename}/demo_{self.demo_num}/sample_{self.sample_count}_depth.png", depth_image, cmap='gray')
                except FileNotFoundError:  # Unable to write image to filesystem
                    pass

            print(f"Sample {self.sample_count}:")
            print(f"\tTimestamp: {timestamp_time:.2f}")
            print(f"\tPosition: {position}")
            print(f"\tOrientation: {orientation}")
            print(f"\tAction: {actions[self.sample_count]}")
            print(f"\tColor Image shape: {color_image.shape}")
            print(f"\tDepth Image shape: {depth_image.shape}\n")
            self.sample_count += 1

    def record(self):
        """Monitor keyboard events to control demonstration recording events."""
        with keyboard.Events() as events:
            for event in events:
                if isinstance(event, keyboard.Events.Press):
                    if event.key == keyboard.Key.enter and not self.recording:
                        self.start_recording()
                    elif event.key == keyboard.KeyCode.from_char('q'):
                        if self.recording:
                            self.stop_recording()
                        else:
                            print("\nQuitting program...")
                            break  # Exit the loop and quit if <q> is pressed when not recording
                    elif event.key == keyboard.KeyCode.from_char('\x03'):
                        if self.recording:
                            self.stop_recording()
                        print("<Ctrl+C> pressed. Quitting program...")
                        exit(0)

    def record_sample_thread(self):
        """Thread to record samples to also allow for monitoring keyboard input"""
        while self.recording:
            self.record_sample()
            time.sleep(self.sample_interval)


def main():
    filename = input("Enter the filename to store the demonstration data in.\n")
    sample_interval = float(input("Enter the sampling interval (in seconds) for data collection.\n"))
    with h5py.File(f"{filename}.hdf5", "w") as file:
        recorder = Recorder(file, sample_interval, store_observations=True)
        print("Press <ENTER> to start recording a demonstration or press <q> to exit the program.")
        recorder.record()
        file.attrs["num_demos"] = recorder.demo_num
        print(f"Total demonstrations recorded: {recorder.demo_num}")


if __name__ == "__main__":
    main()
