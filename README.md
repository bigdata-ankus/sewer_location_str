"str_meter.ipynb" detects distance information at captured image frames of sewerage clips from Seoul Metropolitan Government, South Korea.

Users need to type inputs: execution path(default path) and image file path of captured image frames.
path = 'A:/AA/AAA/', 
image = '00-00000-00'

2nd cell of str_meter.ipynb selects location of distance information within an image frame.

STR(scene-text-recognition) writes distance information(meter in metric unit) in "file.txt":
['000.0', '000.0', '001.7', '007.9', '008.2', '008.7', '015.1', '015.6', '020.0', '024.0'].

Please unzip followings to generate trained files:
sewer_location_str/CRAFT-pytorch/craft_mlt_25k.pth,
sewer_location_str/saved_models/models/train/train_save_part.pth.
