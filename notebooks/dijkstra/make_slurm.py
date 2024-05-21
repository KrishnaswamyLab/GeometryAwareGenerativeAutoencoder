import os
# files_old = ['Hemisphere_none_0','Hemisphere_5_0','Hemisphere_15_0','Hemisphere_50_0','Hemisphere_15_0.1','Hemisphere_15_0.3','SwissRoll_none_0','SwissRoll_5_0','SwissRoll_15_0','SwissRoll_50_0','SwissRoll_15_0.1','SwissRoll_15_0.3','Ellipsoid_15_0.1','Ellipsoid_15_0','Ellipsoid_5_0','Ellipsoid_15_0.3','Ellipsoid_50_0','Ellipsoid_none_0','Torus_15_0.1','Torus_15_0.3','Torus_15_0','Torus_50_0','Torus_5_0','Torus_none_0','Saddle_15_0.1','Saddle_15_0.3','Saddle_15_0','Saddle_50_0','Saddle_5_0','Saddle_none_0']
foldername = '/gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/data/neurips_results/toy/gt/'
prepend = 'cd /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/notebooks/dijkstra/; /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/.conda_envs/autometric/bin/python ground_truth_datasets.py'
# files_new = ['Ellipsoid_15_0','Ellipsoid_15_0.1','Ellipsoid_15_0.3','Ellipsoid_15_0.5','Ellipsoid_15_0.7','Ellipsoid_none_0.1','Ellipsoid_5_0.1','Ellipsoid_10_0.1','Ellipsoid_15_0.1','Ellipsoid_50_0.1','Hemisphere_15_0','Hemisphere_15_0.1','Hemisphere_15_0.3','Hemisphere_15_0.5','Hemisphere_15_0.7','Hemisphere_none_0.1','Hemisphere_5_0.1','Hemisphere_10_0.1','Hemisphere_15_0.1','Hemisphere_50_0.1','Saddle_15_0','Saddle_15_0.1','Saddle_15_0.3','Saddle_15_0.5','Saddle_15_0.7','Saddle_none_0.1','Saddle_5_0.1','Saddle_10_0.1','Saddle_15_0.1','Saddle_50_0.1','Torus_15_0','Torus_15_0.1','Torus_15_0.3','Torus_15_0.5','Torus_15_0.7','Torus_none_0.1','Torus_5_0.1','Torus_10_0.1','Torus_15_0.1','Torus_50_0.1']
files_old = ['Ellipsoid_15_0.1','Hemisphere_15_0.5','Saddle_15_0.3','SwissRoll_5_0','Ellipsoid_15_0.3','Hemisphere_15_0.7','Saddle_15_0','SwissRoll_none_0','Ellipsoid_15_0','Hemisphere_15_0','Saddle_50_0','Torus_15_0.1','Ellipsoid_50_0','Hemisphere_50_0.1','Saddle_5_0','Torus_15_0.3','Ellipsoid_5_0','Hemisphere_50_0','Saddle_none_0.1','Torus_15_0','Ellipsoid_none_0.1','Hemisphere_5_0.1','Saddle_none_0','Torus_50_0','Ellipsoid_none_0','Hemisphere_5_0','SwissRoll_15_0.1','Torus_5_0','Hemisphere_10_0.1','Hemisphere_none_0.1','SwissRoll_15_0.3','Torus_none_0.1','Hemisphere_15_0.1','Hemisphere_none_0','SwissRoll_15_0','Torus_none_0','Hemisphere_15_0.3','Saddle_15_0.1','SwissRoll_50_0']
files_new = [
 'Torus_5_0.1',
 'Torus_15_0.7',
 'Torus_50_0.1',
 'Torus_15_0.5',
 'Torus_10_0.1',
 'Saddle_15_0.7',
 'Saddle_15_0.5',
 'Hemisphere_5_0.1',
 'Hemisphere_50_0.1',
 'Saddle_5_0.1',
 'Hemisphere_10_0.1',
 'Hemisphere_15_0.7',
 'Ellipsoid_15_0.5',
 'Hemisphere_15_0.5',
 'Ellipsoid_10_0.1',
 'Ellipsoid_5_0.1',
 'Ellipsoid_15_0.7',
 'Ellipsoid_50_0.1']
files_new2 = ['Saddle_10_0.1', 'Saddle_50_0.1']
files = [f for f in files_new2 if f not in files_old + files_new]
# files = [f for f in files_new if f not in files_old]
with open('job.txt', 'w') as file:
    for data_name in files:
        mfd, rot_dim, noise  = data_name.split('_')
        if rot_dim == 'none':
            command = f"{prepend} --foldername \"{foldername}\" --mfd {mfd} --noise {noise}"# --rot_dim {rot_dim}"
        else:
            command = f"{prepend} --foldername \"{foldername}\" --mfd {mfd} --noise {noise} --rot_dim {rot_dim}"
        file.write(command + '\n')
print(len(files))

# with open('job.txt', 'w') as file:
#     for data_name in files:
#         mfd, rot_dim, noise  = data_name.split('_')
#         if rot_dim == 'none':
#             command = f"{prepend} --foldername \"{foldername}\" --mfd {mfd} --noise {noise}"# --rot_dim {rot_dim}"
#             file.write(command + '\n')
