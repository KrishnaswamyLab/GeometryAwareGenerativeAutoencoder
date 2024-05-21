import os
# folder = '../../data/gt_geodesic/'
# folder = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/data/neurips_results/toy/'
# files = os.listdir(folder)
files = ['hemisphere_none_0','hemisphere_5_0','hemisphere_15_0','hemisphere_50_0','hemisphere_15_0.1','hemisphere_15_0.3','swissroll_none_0','swissroll_5_0','swissroll_15_0','swissroll_50_0','swissroll_15_0.1','swissroll_15_0.3','ellipsoid_15_0.1','ellipsoid_15_0','ellipsoid_5_0','ellipsoid_15_0.3','ellipsoid_50_0','ellipsoid_none_0','torus_15_0.1','torus_15_0.3','torus_15_0','torus_50_0','torus_5_0','torus_none_0','saddle_15_0.1','saddle_15_0.3','saddle_15_0','saddle_50_0','saddle_5_0','saddle_none_0']
files = [f"{n}.npz" for n in files]
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
