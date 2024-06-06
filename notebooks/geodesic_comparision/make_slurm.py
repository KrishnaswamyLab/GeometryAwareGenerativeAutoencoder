prepend = 'cd /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/dmae/notebooks/geodesic_comparision/; /gpfs/gibbs/pi/krishnaswamy_smita/xingzhi/.conda_envs/geosink/bin/python run_batch.py'


# with open('job.txt', 'w') as file:
#     ae_sweep_id = 'jtpxi61p'
#     disc_sweep_id = 'g7zjiupc'
#     data_names = ['hemisphere_none_0.0','ellipsoid_none_0.0','torus_none_0.0','saddle_none_0.0']

#     for data_name in data_names:
#         command = f"{prepend} --cfg_main_data_name \"{data_name}\" --cfg_main_ae_sweep_id {ae_sweep_id} --cfg_main_disc_sweep_id {disc_sweep_id}"
#         file.write(command + '\n')

#     ae_sweep_id = 'ys48kno0'
#     disc_sweep_id = 'sxdmzefz'

#     data_names = ['ellipsoid_15_0.0', 'ellipsoid_15_0.1', 'ellipsoid_15_0.3', 'ellipsoid_15_0.5', 'ellipsoid_15_0.7', 'ellipsoid_none_0.1', 'ellipsoid_5_0.1', 'ellipsoid_10_0.1', 'ellipsoid_15_0.1', 'ellipsoid_50_0.1', 'hemisphere_15_0.0', 'hemisphere_15_0.1', 'hemisphere_15_0.3', 'hemisphere_15_0.5', 'hemisphere_15_0.7', 'hemisphere_none_0.1', 'hemisphere_5_0.1', 'hemisphere_10_0.1', 'hemisphere_15_0.1', 'hemisphere_50_0.1', 'saddle_15_0.0', 'saddle_15_0.1', 'saddle_15_0.3', 'saddle_15_0.5', 'saddle_15_0.7', 'saddle_none_0.1', 'saddle_5_0.1', 'saddle_10_0.1', 'saddle_15_0.1', 'saddle_50_0.1', 'torus_15_0.0', 'torus_15_0.1', 'torus_15_0.3', 'torus_15_0.5', 'torus_15_0.7', 'torus_none_0.1', 'torus_5_0.1', 'torus_10_0.1', 'torus_15_0.1', 'torus_50_0.1']

#     for data_name in data_names:
#         command = f"{prepend} --cfg_main_data_name \"{data_name}\" --cfg_main_ae_sweep_id {ae_sweep_id} --cfg_main_disc_sweep_id {disc_sweep_id}"
#         file.write(command + '\n')

#     ae_sweep_id = 'ys48kno0'
#     disc_sweep_id = 'rrcxupsf'

#     data_names = ['saddle_10_0.1', 'saddle_50_0.1']

#     for data_name in data_names:
#         command = f"{prepend} --cfg_main_data_name \"{data_name}\" --cfg_main_ae_sweep_id {ae_sweep_id} --cfg_main_disc_sweep_id {disc_sweep_id}"
#         file.write(command + '\n')

    

with open('job.txt', 'w') as file:
    # ae_sweep_id = 'jtpxi61p'
    # disc_sweep_id = 'g7zjiupc'
    # data_names = ['hemisphere_none_0','ellipsoid_none_0','torus_none_0','saddle_none_0']

    # for data_name in data_names:
    #     command = f"{prepend} --cfg_main_data_name \"{data_name}\" --cfg_main_ae_sweep_id {ae_sweep_id} --cfg_main_disc_sweep_id {disc_sweep_id}"
    #     file.write(command + '\n')

    # ae_sweep_id = 'ys48kno0'
    # disc_sweep_id = 'sxdmzefz'

    # data_names = ['ellipsoid_15_0.0', 'ellipsoid_15_0.1', 'ellipsoid_15_0.3', 'ellipsoid_15_0.5', 'ellipsoid_15_0.7', 'ellipsoid_none_0.1', 'ellipsoid_5_0.1', 'ellipsoid_10_0.1', 'ellipsoid_15_0.1', 'ellipsoid_50_0.1', 'hemisphere_15_0.0', 'hemisphere_15_0.1', 'hemisphere_15_0.3', 'hemisphere_15_0.5', 'hemisphere_15_0.7', 'hemisphere_none_0.1', 'hemisphere_5_0.1', 'hemisphere_10_0.1', 'hemisphere_15_0.1', 'hemisphere_50_0.1', 'saddle_15_0.0', 'saddle_15_0.1', 'saddle_15_0.3', 'saddle_15_0.5', 'saddle_15_0.7', 'saddle_none_0.1', 'saddle_5_0.1', 'saddle_10_0.1', 'saddle_15_0.1', 'saddle_50_0.1', 'torus_15_0.0', 'torus_15_0.1', 'torus_15_0.3', 'torus_15_0.5', 'torus_15_0.7', 'torus_none_0.1', 'torus_5_0.1', 'torus_10_0.1', 'torus_15_0.1', 'torus_50_0.1']
    # data_names0 = [n[:-2] for n in data_names if n.endswith('_0.0')]
    # data_names1 = [n for n in data_names if not n.endswith('_0.0')]
    # data_names = data_names0 + data_names1
    # for data_name in data_names:
    #     command = f"{prepend} --cfg_main_data_name \"{data_name}\" --cfg_main_ae_sweep_id {ae_sweep_id} --cfg_main_disc_sweep_id {disc_sweep_id}"
    #     file.write(command + '\n')

    ae_sweep_id = 'ys48kno0'
    disc_sweep_id = 'c5cpmrgi' # rerun failed crashed

    data_names = ['ellipsoid_10_0.1','torus_15_0.7','torus_50_0.1','hemisphere_10_0.1','hemisphere_5_0.1','saddle_5_0.1','hemisphere_15_0.5','hemisphere_50_0.1']
    data_names0 = [n[:-2] for n in data_names if n.endswith('_0.0')]
    data_names1 = [n for n in data_names if not n.endswith('_0.0')]
    data_names = data_names0 + data_names1
    for data_name in data_names:
        command = f"{prepend} --cfg_main_data_name \"{data_name}\" --cfg_main_ae_sweep_id {ae_sweep_id} --cfg_main_disc_sweep_id {disc_sweep_id}"
        file.write(command + '\n')



# with open('job.txt', 'w') as file:
#     ae_sweep_id = 'ys48kno0'
#     disc_sweep_id = 'sxdmzefz'

#     # nan-loss -- reduce alpha and learning rate.
#     data_names = ['hemisphere_15_0.3', 'hemisphere_15_0.3', 'hemisphere_15_0.3',
#        'saddle_15_0.1', 'saddle_50_0.1', 'saddle_50_0.1', 'torus_15_0.5',
#        'torus_15_0.5', 'torus_15_0.5', 'torus_15_0.7', 'torus_15_0.7',
#        'torus_15_0.7', 'torus_10_0.1', 'torus_50_0.1', 'torus_50_0.1']

#     for data_name in data_names:
#         command = f"{prepend} --cfg_main_data_name \"{data_name}\" --cfg_main_ae_sweep_id {ae_sweep_id} --cfg_main_disc_sweep_id {disc_sweep_id}"
#         file.write(command + '\n')

#     # ae_sweep_id = 'ys48kno0'
#     # disc_sweep_id = 'rrcxupsf'

#     # data_names = ['saddle_10_0.1', 'saddle_50_0.1']

#     # for data_name in data_names:
#     #     command = f"{prepend} --cfg_main_data_name \"{data_name}\" --cfg_main_ae_sweep_id {ae_sweep_id} --cfg_main_disc_sweep_id {disc_sweep_id}"
#     #     file.write(command + '\n')

     

