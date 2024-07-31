"str_meter.ipynb" detects distance information at captured image frames of sewerage clips from Seoul Metropolitan Government, South Korea.

Users need to type inputs: execution path(default path) and image file path of captured image frames.
path = 'A:/AA/AAA/', 
image = '00-00000-00'

2nd cell of str_meter.ipynb selects location of distance information within an image frame.

STR(scene-text-recognition) generates distance information(meter in metric unit) in "file.txt".

Please unzip trained files:
sewer_location_str/CRAFT-pytorch/craft_mlt_25k.pth,
sewer_location_str/saved_models/models/train/train_save_part.pth
