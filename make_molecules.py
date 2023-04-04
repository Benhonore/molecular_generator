import torch
import dgl
import networkx as nx
import random
import sys
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



### The purpose of this function is to make a partially connected. This is useful for two reason: 1) Illustrating the
### connectivity (below) and 2) Determining whether the structure is two distinct molecules for filtering.

def make_partial_graph(g):
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

    return new_g

### The purpose of this function is to illustrate the structure of the a newly generated graph, because these graphs are all
### fully connected, just drawing the graph as is, is unhelpful so instead we use this function to make a partially connected
### graph out of the fully connected output graph where edges = bonds.


def draw_graph_connectivity(g):

    partial_g = make_partial_graph(g)
    visualize_graph(partial_g)



### This function makes a fully connected DGL graph from a molecular dataframe. It encodes bond order between (currently)
### hydrogens and the atoms that they are bonded to which we are currently referring to as hard edges. This is on the basis
### that HSQC data is available as well as NMR data and long term we will need to remove this fpr heteroatoms. It also returns
### a mask to mask all edges that are not editable which is sued later in the update graph function.

def make_full_graph(atom_df):
    g = dgl.DGLGraph()
    g.add_nodes(len(atom_df))
    ndata = {'type': torch.as_tensor(atom_df['typeint'].to_numpy(), dtype=torch.int64),
            'shift':torch.as_tensor(atom_df['shift'].to_numpy(), dtype=torch.float32)}
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

    edata= {'distance':torch.as_tensor(edge_type, dtype=torch.int64)}
    g.add_edges(cpl_src, cpl_end)
    g.edata.update(edata)

    return g, np.array(mask)


### Simply, this function makes wha we know to be the real graph - essential for measuring the performance to this method.

def make_real_graph(atom_df):
    g = dgl.DGLGraph()
    g.add_nodes(len(atom_df))
    ndata = {'type': torch.as_tensor(atom_df['typeint'].to_numpy(), dtype=torch.int64), 'shift':torch.as_tensor(atom_df['shift'].to_numpy())}
    g.ndata.update(ndata)


    cpl_src=[]
    cpl_end=[]
    edge_type = []

    for atom in range(len(atom_df)):
        for atom2, con in enumerate(atom_df.iloc[atom]['conn']):
                if atom == atom2:
                    continue
                else:
                    cpl_src.append(atom)
                    cpl_end.append(atom2)

                    edge_type.append(con)

    edata= {'distance':torch.as_tensor(edge_type, dtype=torch.int64)}
    g.add_edges(cpl_src, cpl_end)
    g.edata.update(edata)

    return g



### This function simply takes takes a list of length 'size' and randomly distributes a number 'n' across that list
### in 0s, 1s and 2s. The sum of the output = n and there are no numners bigger than 2. It is designed to randomise valency.

def distribute(n, size):
    lst = [0] * size
    while sum(lst) < n:
        p= random.randint(0, size-1)
        if lst[p] < 2:
            lst[p] +=1
    return lst


### This function checks if a graph that has been generated by the molecule generator is a single molecule or not.

def single_molecule(g):
    rep_g = make_partial_graph(g)
    num_components = nx.number_connected_components(dgl.to_networkx(rep_g).to_undirected())
    return num_components == 1


### This function is written to introduce more logic to the random updating of edges in the molecule search than the more
### original scatter gun approach of randomly distributing the total number of available bonding electrons across all available
### which unsurpisingly does not scale well with molecular size.

### It relies on the make graph and distribute functions but does its own interanl valency checks.

### The starting node is selected randomly, then the electrons available to that node randomly distributed across its own edges.
### The valencies are updated and checked, if the check is failed, the random bond formation step is repeated but if the check
### passes then the next atom is randomly selected and the process repeated. After 20 failed attempts to find a valence-suitable
### update, the bonding structure is reset and the search begins from scratch.


def find_molecules(df, aim, multiple_molecules = False):

    types = {'1':'H', '6':'C', '8':'O'}
    valency_dict = {'6': 4, '8': 2, '1':1}
    graphs = []
    
    last_mol_found = []
    
    count=0

    while len(graphs) < int(aim):
        ### When a valid moelcule is found, the iteration is added to a list. If it has been more that 1000 iterations since the
        ### last valid structrue was found, the search will be terminated on the basis that all molecules have been found.

        if len(last_mol_found) !=0:
            if count - int(last_mol_found[-1]) > 1000:
                print(f'1000 iterations since last molecule found: {len(graphs)} molecules found.')
                break
            else:
                finish = False
        else:
            finish = False
    
        graph, mask = make_full_graph(df)

        ### A dictionary of the availbale valencies of each atom is generated.

        d = {}

        for node, typ in enumerate(graph.ndata['type']):
            if int(typ) == 1:
                continue
            if int(typ) == 6:
                val = 4
            if int(typ) == 8:
                val = 2

            for i, node_index in enumerate(graph.edges()[0]):
                if int(node_index) == node:
                    rtn_node = graph.edges()[1][i]
                    if graph.ndata['type'][rtn_node] == 1:
                        if graph.edata['distance'][i] == 1:
                            val-= 1
            d[node] = val

        
        og_d = d.copy()

        src, dst = graph.edges()
        src = src[mask]
        dst = dst[mask]
        num_edges = len(src)

        real_graph = False

        original = graph.edata['distance'][mask]
        
    
        while not real_graph:
            
            if finish:
                break
    
            if sum(list(d.values())) == 0:
            
                    real_graph = True
                    break
    
            c=0
    
            node = random.choice(list(d.keys()))
    
            current = graph.edata['distance'][mask]
    
    
            if d[node] > 0:
        
        
                val = False
        
                while not val:
                    temp = graph.edata['distance'][mask]
            
                    temp_src = temp[src==node]
                    temp_dst = temp[dst==node]
    
            
                    temp_src[temp_src==0] = torch.tensor(distribute(d[node], len(temp_src[temp_src==0])))
                    temp_dst = temp_src
                    temp[src==node] = temp_src
                    temp[dst==node] = temp_dst
        
                    count+=1
            
                    if len(last_mol_found) !=0:
                        if count - int(last_mol_found[-1]) > 1000:
                            print()
                            print(f'1000 iterations since last molecule found: {len(graphs)} molecules found.')
                            finish=True
                            break
        
        
                    graph.edata['distance'][mask] = temp  # Update bond order
        
                
                    if all ([sum(graph.edata['distance'][graph.edges()[0]==hatom]) <= 
                             valency_dict[str(int(graph.ndata['type'][hatom]))] for hatom in d]):  # Check to see if new bond order works
            
                        for hatom in d:
                                    d[hatom] = int(valency_dict[str(int(graph.ndata['type'][hatom]))] - 
                                                   sum(graph.edata['distance'][graph.edges()[0]==hatom]))   # Update valency dictionary
                
                        #print(d)
                        val = True
                
                
                    else:
                        #print('bad valency')
                        c+=1
                        #print(c)
                        graph.edata['distance'][mask] = current
                
                        if c > 20:
                    
                            graph.edata['distance'][mask] = original
                            d = og_d
                
                        continue
                    
            else:
                continue
        
        
        if finish:
            break
        
        if len(graphs) == 0:
            
            if multiple_molecules:
                graphs.append(graph)
                sys.stdout.write('\rmolecules found: %d' %len(graphs))
                sys.stdout.flush()
                last_mol_found.append(count)
            
            else:
                if single_molecule(graph):
                    graphs.append(graph)
                    sys.stdout.write('\rmolecules found: %d' %len(graphs))
                    sys.stdout.flush()
                    last_mol_found.append(count)

        
        if len(graphs) != 0:

            if multiple_molecules:
                if all ([not torch.equal(i.edata['distance'], graph.edata['distance']) for i in graphs]):
                
                    graphs.append(graph)
                    sys.stdout.write('\rmolecules found: %d' %len(graphs))
                    sys.stdout.flush()
                    last_mol_found.append(count)
            else:
                if single_molecule(graph) and all ([not torch.equal(i.edata['distance'], graph.edata['distance']) for i in graphs]):
                    
                    graphs.append(graph)
                    sys.stdout.write('\rmolecules found: %d' %len(graphs))
                    sys.stdout.flush()
                    last_mol_found.append(count)
    
    return graphs, last_mol_found, count
