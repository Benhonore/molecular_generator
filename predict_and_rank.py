import sys
sys.path.append('/Users/bh17536/work_area/imp_core/imp_core/')

from model.gtn_model import GTNmodel as gtn
from model.GTN_modules import scaling as scl
from torch.utils.data import DataLoader
from model.GTN_modules import loss as lossfn

import torch
import numpy as np
import pandas as pd
import dgl
import make_molecules as mm

def collate(samples):

        graphs = samples
        if len(graphs) == 1:
                batched_graph = dgl.batch([graphs[0]])#, graphs[0]])
        else:
                batched_graph = dgl.batch(graphs)

        return batched_graph
    
def mae(x, y):
    if type(x) == np.float64:
        y1 = np.float(y)
        return x-y1
    else:
        return sum(abs(x-y))/len(x)

train_model=gtn(model_args={'targetflag':['HCS', 'CCS']})
train_model.load_model('/Users/bh17536/work_area/IMPRESSION_models/2D_IMPRESSION_without_path_len_OPT_checkpoint.torch')
#train_model.load_model('/home/benhonore/DT45_2D_IMPRESSION_OPT_checkpoint.torch')

def rank_structures(mols): # Current botch: zero all of the 3D relevant edge features 

    indices=['nmr_type', 'coupling', 'path_len']
    for i in range(len(mols)):
        for index in indices:
            mols[i].edata[index] = torch.zeros(len(mols[i].edata['distance']), dtype=torch.int)

    x_loader = DataLoader(mols, batch_size=2, collate_fn=collate)
    preds=train_model.predict(x_loader)

    h_error = []
    c_error = []
    h_d = {}
    c_d = {}

    for pred in preds: # Loop through output graphs, pulling out the predictions for H & C

        prediction, truth = lossfn.get_outputs(pred, 'HCS')
        pred_y = scl.descale_and_denorm(prediction.numpy().squeeze(), train_model.args['scaling']['HCS'])
        h_error.append(mae(pred_y, truth.numpy()))
        h_d[mae(pred_y, truth.numpy())] = pred
        prediction, truth = lossfn.get_outputs(pred, 'CCS')
        pred_y = scl.descale_and_denorm(prediction.numpy().squeeze(), train_model.args['scaling']['CCS'])
        c_error.append(mae(pred_y, truth.numpy()))
        c_d[mae(pred_y, truth.numpy())] = pred

    if c_error.index(np.min(c_error)) == h_error.index(np.min(h_error)):
         print('Final Structure..')
         mm.draw_graph_connectivity(preds[c_error.index(np.min(c_error))]) # IF the lowest error agrees for H & C
    
    df =pd.DataFrame()
    df['graphs'] = preds
    df['h_error'] = h_error
    df['c_error'] = c_error

    return df   # A dataframe of graph | proton_error | carbon_error is returned


def evaluate(df, real_graph):

    right_mol_h = False
    right_mol_c = False

    for i in range(len(df.sort_values(by=['h_error']))):
        if list(df.sort_values(by=['h_error']).iloc[i]['graphs'].edata['distance'])== list(real_graph.edata['distance']):
            print(f'PROTON ranked: {i+1}')
            h=i+1
        
    for i in range(len(df.sort_values(by=['c_error']))):
        if list(df.sort_values(by=['c_error']).iloc[i]['graphs'].edata['distance'])== list(real_graph.edata['distance']):
            print(f'CARBON ranked: {i+1}')
            c=i+1

    if mm.isomorphic(df.sort_values(by=['h_error']).iloc[0]['graphs'], real_graph):
        right_mol_h = True
    if mm.isomorphic(df.sort_values(by=['c_error']).iloc[0]['graphs'], real_graph):
        right_mol_c = True

    return h, c, right_mol_h, right_mol_c
