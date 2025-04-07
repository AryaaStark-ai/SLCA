import json
from matplotlib.image import imread
import numpy as np
import os

from collections import Counter

from measure import *
from slca import *

class DataLoader:
    def __init__(self, data_folder, desired_res):
        self.data_folder = data_folder
        
        self.desired_res = desired_res
        self.human_rt = self.load_human_data('all_rts')
        self.stim = self.load_stim()
        self.human_coords = self.get_human_gaze()

    def load_human_data(self, file):
        with open(f'{self.data_folder}/{file}.json', 'r') as f:
            data = json.load(f)
        return data

    def get_human_gaze(self):
        human_coords = self.load_human_data('all_coords')
        data = {}
        for stim in human_coords:
            data[stim] = {}

            for participant in human_coords[stim]:
                if stim in self.stim:
                    gazepoints = human_coords[stim][participant]
                    if len(gazepoints) > 0:
                        lens = self.human_rt[stim][participant]
                        gaze = np.zeros(self.stim[stim]['map'].shape)
                        for i in range(len(gazepoints)):
                            y, x = gazepoints[i]
                            try:
                                gaze[int(x / self.x_decrease)][int(y / self.y_decrease)] += lens[i]
                            except:
                                pass
                        gaze = (gaze - np.min(gaze)) / (np.max(gaze) - np.min(gaze))
                        data[stim][participant] = np.array(gaze, dtype=np.float32)
        return data

    def load_stim(self):
        pics = {}
        for pic in self.human_rt:
            im_path = f'{self.data_folder}/smaps/eml_net/{pic[:-4]}_smap.png'
            if os.path.isfile(im_path):
                eml_net = imread(im_path)

                if self.desired_res[0] > eml_net.shape[1] or self.desired_res[1] > eml_net.shape[0]:
                    raise Exception('Desired saliency map resolution should be less than original resolution. Please change the parameter desired_res in the file slca_static_parameters.json')
                
                self.y_decrease = round(eml_net.shape[0] / self.desired_res[1])
                self.x_decrease = round(eml_net.shape[1] / self.desired_res[0])
                
                eml_net = eml_net[::self.y_decrease, ::self.x_decrease]
                eml_net = (eml_net - np.min(eml_net)) / (np.max(eml_net) - np.min(eml_net))
                pics[pic] = {'map': eml_net, 'salient': len(eml_net[eml_net > 0.6])}
        return pics          
           
class ParamsLoader:
    def __init__(self, parameter_folder):
        self.parameter_folder = parameter_folder
        self.all_params = {'dt_t': 0, 'leak': 1, 'competition': 2, 'self_excit': 3, 'w_cross': 4, 'w_input': 5, 'offset': 6, 'noise_sd': 7, 
                           'threshold': 8, 'threshold_change': 9}
        
    def _parse_params(self, params):
        return params
        
    def load(self, filename):
        self.filename = filename
        file_path = f'{self.parameter_folder}/{filename}.json'
        if not os.path.isfile(file_path):
            raise Exception(f'Please check the parameters folder {self.parameter_folder}! It should contain file {self.filename}.json. Either file is absent, or the name of the file is incorrect, or there is no such file in the folder.')
        else:
            try:
                with open(file_path, 'r') as f:
                    params = json.load(f)
                    return self._parse_params(params)
            except:
                raise Exception(f'Error while reading file {file_path} in the folder {self.parameter_folder}. Please check whether the file follows correct json standards.')
           
class GA_ParamsLoader(ParamsLoader):
    def __init__(self, parameter_folder):
        super().__init__(parameter_folder)
        self.available_metrics = {'ks': {'descr': 'Kolmogorov-Smirnov, a temporal metric', 'method': KolmogorovSmirnov}, 
                                  'aj': {'descr': 'AUC-Judd, a spatial metric', 'method': AucJudd}
                                  }
    
    def _parse_params(self, params):
        n_generations, gen_size, model_type, participants, metrics = params.values()
        if gen_size % 3 != 0:
            raise Exception(f'The amount of the descendants in one generation should be a multiple of 4. Please check the parameter gen_size in the file {self.filename}.json')
        
        if model_type == 'local':
            lca_model = SLCA_local
        elif model_type == 'global':
            lca_model = SLCA_global
        else:
            raise Exception(f'Please specify one of the available types of the SLCA model: local or global (file {self.filename}.json, parameter model_type')
        
        n_metrics = 0
        metric_methods = {}
        for metric_group_1 in metrics:
            metric_methods[metric_group_1] = {}
            for metric_group_2 in metrics[metric_group_1]:
                metric_methods[metric_group_1][metric_group_2] = {}
                #if user specified an unknown metric in the {self.filename}.json file
                for metric in metrics[metric_group_1][metric_group_2]:
                    if metric not in self.available_metrics:
                        raise Exception(get_metric_exception_row(self.available_metrics, f'Only the following metrics are available for the parameter metrics in the file {self.filename}.json:'))
                    
                    metric_methods[metric_group_1][metric_group_2][metric] = self.available_metrics[metric]['method']                
                    n_metrics += 1
        
        #if user did not specify any metric in the {self.filename}.json file
        if n_metrics == 0:
            raise Exception(get_metric_exception_row(self.available_metrics, f'Please specify at least one of the following metrics in the file {self.filename}.json:'))
        
        if len(participants) == 0:
            participants = list(range(1,47))
            
        return n_generations, gen_size, model_type, participants, metric_methods, n_metrics, lca_model
        
class SLCA_ParamsInitLoader(ParamsLoader):
    def __init__(self, parameter_folder):
        super().__init__(parameter_folder)
        
    def _parse_params(self, param_sets):
        for _, params in param_sets.items():
            
            if len(params) > len(set(params.keys())):
                param_counter = Counter(list(params.leys()))
                param_multiple = {p: param_counter[p] for p in param_counter if param_counter[k]>0}
                raise Exception(f'Parameter(s) {param_multiple} is specified multiple times in file {self.filename}.json')
    
        return [{self.all_params[p]: param_set[p] for p in param_set if p in self.all_params} for param_set in param_sets.values()]
        
class SLCA_ParamsRangeLoader(ParamsLoader):
    def __init__(self, parameter_folder):
        super().__init__(parameter_folder)
        self.default_range = {"dt_t": [0.0, 1.0],
                              "leak": [0.1, 0.5],
                              "competition": [0.0005, 1.0],
                              "self_excit": [0.05, 0.5],
                              "w_input": [0.1, 0.9],
                              "w_cross": [0.2, 1.0],
                              "offset": [0.0, 10.0],
                              "noise_sd": [0.2, 5.0],
                              "threshold": [2.0, 25.0],
                              "threshold_change": [0.0, 5.0]}
     
    def _parse_params(self, params):
        if len(params) > len(set(params.keys())):
            param_counter = Counter(list(params.leys()))
            param_multiple = {p: param_counter[p] for p in param_counter if param_counter[k]>0}
            raise Exception(f'Parameter(s) {param_multiple} is specified multiple times in file {self.filename}.json')
        for param in params:
            if param in self.default_range:
                self.default_range[param] = params[param]
        return self.default_range
        
class SLCA_ParamsFixedLoader(ParamsLoader):
    def __init__(self, parameter_folder):
        super().__init__(parameter_folder)
        
    def _parse_params(self, params):
        if len(params) > len(set(params.keys())):
            param_counter = Counter(list(params.leys()))
            param_multiple = {p: param_counter[p] for p in param_counter if param_counter[k]>0}
            raise Exception(f'Parameter(s) {param_multiple} is specified multiple times in file {self.filename}.json')
        return {self.all_params[p]: params[p] for p in params if p in self.all_params}
        
def get_metric_exception_row(available_metrics, exception_start):
    exception_row = exception_start
    for metric in self.available_metrics:
        exception_row += f"\n{metric} - {self.available_metrics[metric]['descr']}"
    return exception_row