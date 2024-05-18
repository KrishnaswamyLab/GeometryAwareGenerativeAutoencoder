import os
# folder = '../../data/gt_geodesic/'
folder = '../../data/swiss_roll_wide_geod/'
files = os.listdir(folder)
prepend = 'cd /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/notebooks/datasets_with_geodesic/; /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/.conda_envs/geosink/bin/python neg_samp.py'

# Open a file named 'job.txt' in write mode
with open('job.txt', 'w') as file:
    # Loop through each data name
    for data_name in files:
        # if data_name.startswith('torus') or data_name.startswith('saddle') or data_name.startswith('ellipsoid'):
            # Format the command string
            command = f"{prepend} --data_name {data_name}"
            # Write the command to the file followed by a newline character
            file.write(command + '\n')
