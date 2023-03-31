import torch
import dgl
import networkx as nx
import random

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


### This function draws the DGL graph via networkX

def visualize_graph(g):

    color = {'1':'grey', '6':'blue', '8':'red'}
    atom_type = {'1':'H', '6':'C', '8':'O'}

    color_map = []
    label_dict = {}
    types = []

    for i, typ in enumerate(g.ndata['type']):
        color_map.append(color[str(int(typ))])
        label_dict[i] = atom_type[str(int(typ))] + str(i)
        if typ > 1:
            size = typ -2
        else:
            size = 1
        types.append(size)


    G = dgl.to_networkx(g)

    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), node_size=[v*200 for v in types], labels = label_dict, with_labels=True,
                     node_color=color_map, cmap="Set2")
    plt.show()


### This function makes a fully connected DGL graph from a molecular dataframe. It encodes bond order between (currently) 
### hydrogens and the atoms that they are bonded to which we are currently referring to as hard edges. This is on the basis
### that HSQC data is available as well as NMR data and long term we will need to remove this fpr heteroatoms. It also returns
### a mask to mask all edges that are not editable which is sued later in the update graph function.

def make_full_graph(atom_df):
    g = dgl.DGLGraph()
    g.add_nodes(len(atom_df))
    ndata = {'type': torch.as_tensor(atom_df['typeint'].to_numpy(), dtype=torch.int64)}
    g.ndata.update(ndata)
    
    
    cpl_src=[]
    cpl_end=[]
    
    mask = []
    
    edge_type = []
    
    for atom in range(len(atom_df)):
        for atom2 in range(len(atom_df)):
                if atom == atom2:
                    continue
        
                if atom_df.iloc[atom]['typestr'] =='H' or atom_df.iloc[atom2]['typestr']=='H':
                    
                    cpl_src.append(atom)
                    cpl_end.append(atom2)              
                    
                    edge_type.append(atom_df.iloc[atom]['conn'][atom2])
                    
                    mask.append(False)
                    
                else:
                    cpl_src.append(atom)
                    cpl_end.append(atom2)
                
                
                    edge_type.append(0)
                    mask.append(True)
            
    edata= {'bond_order':torch.as_tensor(edge_type, dtype=torch.int64)}
    g.add_edges(cpl_src, cpl_end)
    g.edata.update(edata)
    
    return g, np.array(mask)




### This function simply takes takes a list of length 'size' and randomly distributes a number 'n' across that list
### in 0s, 1s and 2s. The sum of the output = n and there are no numners bigger than 2. It is designed to randomise valency.

def distribute(n, size):
    lst = [0] * size
    while sum(lst) < n:
        p= random.randint(0, size-1)
        if lst[p] < 2:
            lst[p] +=1
    return lst



### This function actively updates the graph edges - it takes in the graph and the mask that removes all hard edges or 
### edges that are not to be edited. 

### First, it works out the available bonding electrons of each carbon and heteroatom - if there are already hard edges to
### that atom (in this case, hydrogens bonded), then it will subtract these from the valency - it returns a dictionary of 
### available valencies although currently, we only really use the sum of the that rather than atom by atom. This is a 
### constraint that could be added further down the line.

### It then uses the distribute function to randomly distribute bonds across the masked edge index (in other words, across
### the available edges). I've done a bit of a fudge at the moment to account for the fact that there are two edges for each 
### atom interaction (eg. atom1 -> atom2 AND atom2 -> atom1). These degenerate edges are not necessarily indexed consistently
### which is annoying so what you see below is my temporary fix. 

### A problem was encountered with the fact that a new graph MUST be made to ensure that proper updating of the edges takes
### place further down the line. A new graph could be made from scratch but I've found a nifty way of replicating the graph
### while also making a new graph so thta their properties are not tied to eachother. I'm not 100% conviced by it though and
### the origin is ChatGPT so keep an eye out for any strange errors and I can easily remedy this by just making the new graph
### from scratch.

def update_g(g, mask):
    d = {}

    for node, typ in enumerate(g.ndata['type']):
        if int(typ) == 1:
            continue
        if int(typ) == 6:
            val = 4
        if int(typ) == 8:
            val = 2

        for i, node_index in enumerate(g.edges()[0]):
            if int(node_index) == node:
                rtn_node = g.edges()[1][i]
                if g.ndata['type'][rtn_node] == 1:
                    if g.edata['bond_order'][i] == 1:
                        val-= 1
        d[node] = val


    src, dst = g.edges()
    src = src[mask]
    dst = dst[mask]
    num_edges = len(src)
    
    half_edges = distribute(int(0.5*sum(d)), int(0.5*num_edges))
    edges = []
    new_edges = []
    c=0
    for i in range(len(g.edges()[0][mask])):
        if [int(g.edges()[1][mask][i]), int(g.edges()[0][mask][i])] not in edges:
            edges.append([int(g.edges()[0][mask][i]), int(g.edges()[1][mask][i])])
            new_edges.append(half_edges[c])
            c+=1
        else:
            index = edges.index([int(g.edges()[1][mask][i]), int(g.edges()[0][mask][i])])
            new_edges.append(half_edges[index])
        
    
    g.edata['bond_order'][mask] = torch.tensor(new_edges)
    
    return g




### This function simply checks that the valencies of the graph make chemical sense. It loops through, pulls out the atom 
### type at each node and asserts that its valency is valid.

str_types = {'1':'H', '6':'C', '8':'O'}

def check_valency(g):
    # Create a dictionary of valency constraints for each element
    valency_dict = {'C': 4, 'O': 2, 'H':1}
    # Calculate the valency of each node
    valencies = {}
    for i, node in enumerate(g.ndata['type']):
        symbol = str_types[str(int(node))]
        atom = symbol + '_' + str(i)
        #print(symbol)
        # Hydrogen is counted for at the moment
        if symbol != ' H': 
            bond_order = 0
            # Loop through edges, adding to bond order
            for index, j in enumerate(g.edges()[0]):
                if int(j) == i:
                    bond_order += int(g.edata['bond_order'][index])
                    
        else:
            bond_order = 1
        
        valencies[atom] = bond_order
    
    for symbol, valency in valencies.items():
        if valency != valency_dict[symbol.split('_')[0]]:
            return False
    return True



### The purpose of this function is to illustrate the structure of the a newly generated graph, because these graphs are all
### fully connected, just drawing the graph as is, is unhelpful so instead we use this function to make a partially connected
### graph out of the fully connected output graph where edges = bonds.

def draw_graph_connectivity(g):
    new_g = dgl.DGLGraph()
    new_g.add_nodes(len(g.nodes()))
    new_g.ndata['type'] = g.ndata['type']

    src = []
    dst = []

    for node in range(len(g.nodes())):
        for i, edge_node in enumerate(g.edges()[0]):
            if edge_node == node:
                if g.edata['distance'][i] != 0:
                    src.append(node)
                    dst.append(g.edges()[1][i])
                
    new_g.add_edges(src, dst)

    visualize_graph(new_g)

