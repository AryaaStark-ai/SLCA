import argparse
from genetic import *
from slca import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='local', help='local or global model')
    parser.add_argument('--participant', '-p', type=int, default=0, help='participant number')
    parser.add_argument('--data', '-d', type=str, default='data', help='path to data folder')
    args = parser.parse_args()
    model_type, participant, data_folder = args.model, args.participant, args.data
        
    n_units = 8160
    time = 700
    project_path =  r"/content/drive/MyDrive/slca-main-pavan"
    data_folder = f'{project_path}/data'
    stim_path = f'{project_path}/data/smaps/eml-net1'
    human_rt_path = f'{project_path}/data/all_rts.json'
    human_coords_path = f'{project_path}/data/all_coords.json'
    gens_num = 2
    params = np.zeros(shape=(gens_num, 12, 8))
    
    if model_type == 'local':
        params_first = np.array([[0.256, 0.5, 0.37, 0.64, 0.097, 0.31, 1.043, 0.465],
                                  [0.256, 0.9, 0.25, 0.1, 0.097, 0.31, 3.0, 0.465],
                                  [0.256, 0.8, 0.25, 0.1, 0.097, 0.31, 4.0, 0.2]])
        params_diap = ((0.1, 0.5), (0.0005, 1.0), (0.05, 0.5), (0.1, 0.9), (0.2, 1.0), (0.0,10.0), (0.2,5.0), (0.05, 1.5))
        
    elif model_type == 'global':
        params_first = np.array([[0.4, 0.024, 0.4, 0.1, 1.0, 0.1, 1., 0.178],
                                  [0.3, 0.003, 0.4, 0.2, 0.8, 0.1, 2., 0.5],
                                  [0.25, 0.01, 0.4, 0.1, 0.9, 0.2, 3., 0.3]])
        params_diap = ((0.1, 0.5), (0.0005, 0.1), (0.05, 0.5), (0.1, 0.9), (0.2, 1.0), (0.0,10.0), (0.2,5.0), (0.05, 1.5))

    
    ga = GA(lca_model=model_type, time=time, n_units=n_units, stim_path=stim_path, human_rt_path=human_rt_path,
            human_coords_path=human_coords_path, params_diap=params_diap, part=participant)
    params[0] = ga.first_gen()
    params[0][:3] = params_first
                
    for g in range(0, gens_num):
        print(f'GENERATION {g}/{gens_num}')
        try:
        # Call next_gen with the stored parameters
            params[g] = ga.next_gen(g, params[g - 1])   
            print(f'GENERATION {g} RES {list(params[g])}')
        except Exception as e:
            print("!!! Exception", str(e))
    