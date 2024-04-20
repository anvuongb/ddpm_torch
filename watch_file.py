import os
import glob
import time

search_dir = "/home/anvuong/Desktop/codes/ddpm_torch/sampling_images/mnist"

last_file = ""

while True:
    files = glob.glob(search_dir + "/*.png")
    files.sort(key=lambda x: os.path.getmtime(x))

    if files[-1] != last_file:
        s = files[-1].split("/")[-1]
        print(f"File updated, found {s}, updating symlink")
        if os.path.exists(search_dir + "/latest.png"):
            os.remove(search_dir + "/latest.png")
        
        os.symlink(files[-1], search_dir + "/latest.png")
        last_file = files[-1]

    time.sleep(180)