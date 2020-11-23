datacollection.py collects data from OpenBCI lsl steam, and saves them to a desired output

These paramaters are ment to be adjused depending on the goal:
channels = 8  # Num of headset channels
samples = 375  # 25 samples ~= 1 second
output_folder = 'nothing'  # Name of folder with desired action
trial = trials.left_right()  # Trial to view when collecting data.

You may also commet out this line if you want to have the unadjusted array:
data = data[:,:,:60]  # Trimms to 60hz