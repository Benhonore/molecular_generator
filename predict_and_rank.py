import sys
sys.path.append('/home/benhonore/gtn_imp_core/imp_core')

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
    return sum(abs(x-y))/len(x)

train_model=gtn(model_args={'targetflag':['HCS', 'CCS']})
train_model.load_model('/home/benhonore/2D_IMPRESSION_without_path_len_OPT_checkpoint.torch')
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
    indices = []

    for pred in range(len(preds)): # Loop through output graphs, pulling out the predictions for H & C

        prediction, truth = lossfn.get_outputs(preds[pred], 'HCS')
        pred_y = scl.descale_and_denorm(prediction.numpy().squeeze(), train_model.args['scaling']['HCS'])
        indices.append(pred)
        h_error.append(mae(pred_y, truth.numpy()))
        prediction, truth = lossfn.get_outputs(preds[pred], 'CCS')
        pred_y = scl.descale_and_denorm(prediction.numpy().squeeze(), train_model.args['scaling']['CCS'])
        c_error.append(mae(pred_y, truth.numpy()))

    if c_error.index(np.min(c_error)) == h_error.index(np.min(h_error)):
         print('Final Structure..')
         mm.draw_graph_connectivity(preds[c_error.index(np.min(c_error))]) # IF the lowest error agrees for H & C
    
    h_d = {}
    hcopy = h_error.copy()
    h_error.sort()

    for i, error in enumerate(h_error):
         index = hcopy.index(error) # This just finds the index in the of this error in the original list of errors
         h_d[i] = [error, preds[index]] # The rank (i) is the dictionary key - the values are error and graph

    c_d = {}
    ccopy = c_error.copy()
    c_error.sort()

    for i, error in enumerate(c_error):
         index = ccopy.index(error)
         c_d[i] = [error, preds[index]]
 

    return h_d, c_d # Ranked dictionaries for proton and carbon are returned
