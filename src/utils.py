from collections import defaultdict
import numpy as np

import torch
from torch_geometric.data import Data

from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder


# convert each molecule to a graph 
# nodes are atoms
# edges are bonds

def smiles2geodata(smile, y, node_features_dict, edge_features_dict):
    molecule = Chem.MolFromSmiles(smile)
    
    atomic_number = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    aromaticity = [int(atom.GetIsAromatic()) for atom in molecule.GetAtoms()]
    num_bonds = [atom.GetDegree() for atom in molecule.GetAtoms()]
    bonded_hydrogens = [atom.GetTotalNumHs() for atom in molecule.GetAtoms()]
    hybridization = [atom.GetHybridization().real for atom in molecule.GetAtoms()]
    
    node_keys_features = [f"{atomic}_{aromatic}_{bonds}_{hydrogen}_{hybrid}" 
                        for atomic, aromatic, bonds, hydrogen, hybrid 
                        in zip(atomic_number, aromaticity, num_bonds, bonded_hydrogens, hybridization)]
    
    edge_key_features = []
    for bond in molecule.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()
        in_ring = int(bond.IsInRing())
        conjugated = int(bond.GetIsConjugated())
        stereo = int(bond.GetStereo())
        
        edge_key_features.append(f"{bond_type:.1f}_{in_ring:.1f}_{conjugated:.1f}_{stereo:.1f}") 
    
    nodes_features = torch.tensor(np.array([node_features_dict[x] for x in node_keys_features]), dtype=torch.float32)
    edges_features = torch.tensor(np.array([edge_features_dict[x] for x in edge_key_features]), dtype=torch.float32)  
    
    edges = get_edge_indices(molecule)
    
    # data class that has the dataset - edge features with edge index
    geo_dp = Data(x=nodes_features, edge_index=edges, edge_attr=edges_features, y=y)
    
    return geo_dp


# creates the dictionaries 
def get_atom_features(smile_list):
    #nodes
    atomic_number = []
    aromaticity = []
    num_bonds = []
    bonded_hydrogens = []
    hybridization = []
    
    #edges
    bond_type = []
    in_ring = []
    conjugated = []
    stereo = []
    
    for smi in smile_list:
        molecule = Chem.MolFromSmiles(smi)
        
        #nodes
        atomic_number.extend([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
        aromaticity.extend([int(atom.GetIsAromatic()) for atom in molecule.GetAtoms()])
        num_bonds.extend([atom.GetDegree() for atom in molecule.GetAtoms()])
        bonded_hydrogens.extend([atom.GetTotalNumHs() for atom in molecule.GetAtoms()])
        hybridization.extend([atom.GetHybridization().real for atom in molecule.GetAtoms()])
        
        #edges
        for bond in molecule.GetBonds():
            bond_type.extend([bond.GetBondTypeAsDouble()])
            in_ring.extend([int(bond.IsInRing())])
            conjugated.extend([int(bond.GetIsConjugated())])
            stereo.extend([int(bond.GetStereo())])
            
    #nodes
    atomic_set = list(set(atomic_number))
    codificador_atomic = OneHotEncoder()
    codificador_atomic.fit(np.array(atomic_set).reshape(-1,1))
    
    # can modify features - more representative to molecular properies- boiling point etc. --- number of electron the molecule has? (atom.getBoilgPoint)
    # atomic mass, hydrogen bond acceptors
    # Daniel's powerpoint --- in box---- google in RDkit
    aromatic_set = list(set(aromaticity))
    codificador_aromatic = OneHotEncoder()
    codificador_aromatic.fit(np.array(aromatic_set).reshape(-1,1))
    
    bonds_set = list(set(num_bonds))
    codificador_bonds = OneHotEncoder()
    codificador_bonds.fit(np.array(bonds_set).reshape(-1,1))
    
    hydrogen_set = list(set(bonded_hydrogens))
    codificador_hydrogen = OneHotEncoder()
    codificador_hydrogen.fit(np.array(hydrogen_set).reshape(-1,1))   
    
    hybrid_set = list(set(hybridization))
    codificador_hybrid = OneHotEncoder()
    codificador_hybrid.fit(np.array(hybrid_set).reshape(-1,1))
    
    #edges
    bond_type_set = list(set(bond_type))
    codificador_bond_type = OneHotEncoder()
    codificador_bond_type.fit(np.array(bond_type_set).reshape(-1,1))
    
    in_ring_set = list(set(in_ring))
    codificador_in_ring= OneHotEncoder()
    codificador_in_ring.fit(np.array(in_ring_set).reshape(-1,1))
    
    conjugated_set = list(set(conjugated))
    codificador_conjugated= OneHotEncoder()
    codificador_conjugated.fit(np.array(conjugated_set).reshape(-1,1))
    
    stereo_set = list(set(stereo))
    codificador_stereo= OneHotEncoder()
    codificador_stereo.fit(np.array(stereo_set).reshape(-1,1))

    features_dict = defaultdict(list)
    edge_features_dict = defaultdict(list)
    
    for atom, aromatic, bonds, hydrogen, hybrid in zip(atomic_number, aromaticity, num_bonds, bonded_hydrogens, hybridization):
        node_key_features_combined = f"{atom}_{aromatic}_{bonds}_{hydrogen}_{hybrid}"
        
        atomic_feature  = codificador_atomic.transform([[atom]]).toarray()[0]
        aromatic_feature = codificador_aromatic.transform([[aromatic]]).toarray()[0]
        bonds_feature = codificador_bonds.transform([[bonds]]).toarray()[0]
        hydrogen_feature = codificador_hydrogen.transform([[hydrogen]]).toarray()[0]
        hybrid_feature = codificador_hybrid.transform([[hybrid]]).toarray()[0]
        
        feature_node = np.concatenate((atomic_feature, aromatic_feature, bonds_feature, hydrogen_feature, hybrid_feature))
        features_dict[node_key_features_combined] = feature_node
    
    for bond, ring, conjugat, ster in zip(bond_type, in_ring, conjugated, stereo):
        edge_key_features_combined = f"{bond:.1f}_{ring:.1f}_{conjugat:.1f}_{ster:.1f}" 
        
        bond_feature = codificador_bond_type.transform([[bond]]).toarray()[0]
        ring_feature = codificador_in_ring.transform([[ring]]).toarray()[0]
        conjugated_feature = codificador_conjugated.transform([[conjugat]]).toarray()[0]
        stereo_feature = codificador_stereo.transform([[ster]]).toarray()[0]   
            
        feature_edge = np.concatenate((bond_feature, ring_feature, conjugated_feature, stereo_feature))
        edge_features_dict[edge_key_features_combined] = feature_edge
        
    
    return features_dict, edge_features_dict


# creating how nodes are connected in the graphs 
# edge index - 2xnumber of atoms (nodes)
def get_edge_indices(molecule):
    
    edges =[]
    for bond in molecule.GetBonds():
        edges.append((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))
        
    edges = [[x[0] for x in edges],[x[1] for x in edges]]
    
    return torch.tensor(edges,dtype=torch.long)



# lista_smile = ['Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14', 
#             'COc1cc(OC)c(cc1NC(=O)CSCC(=O)O)S(=O)(=O)N2C(C)CCc3ccccc23', 
#             'CS(=O)(=O)c1ccccc1C(=O)NC[C@@H](O)CN2CCC(CC2)Oc3ccc(Cl)c(Cl)c3', 
#             'CC(C)N(CCC(C(=O)N)(c1ccccc1)c2ccccn2)C(C)C',
#             'Cc1cc(CCC2CCN(CC2)S(=O)(=O)CC3(CCOCC3)N(O)C=O)c(C)cn1'
#             ]



# features = get_atom_features(lista_smile)

# print(features) 
# num_keys = len(features)
# print("Number of keys in the dictionary:", num_keys)
# value_dims = [len(value) for value in features.values()]
# print("Dimensions of dictionary values:", value_dims)
