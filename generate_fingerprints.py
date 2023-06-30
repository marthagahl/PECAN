import numpy as np
import pickle
import sys
import rdkit
from chem import SM_Chem
chem = SM_Chem()


smile_path = sys.argv[1] 
out_path = sys.argv[2]
out_name = sys.argv[3]


def getFP(smile):
    return chem.get_binary(smile)


def getArrays(smile_list):
    fp_list = []
    errors = 0
    error_ids = []
    for smile in smile_list:
        try:
            fp = getFP(smile[1])            
        except:
            errors+=1
            error_ids.append(smile[0])
            continue
    
        fp_list.append(np.array(fp))
        
    return np.array(fp_list), errors, error_ids


def savePickle(fp_array, out_path, out_name):
    save_path = str(out_path) + str(out_name)
    with open(save_path, 'wb') as f:
        pickle.dump(fp_array, f, protocol=pickle.HIGHEST_PROTOCOL)


smiles = []
smile_file = open(smile_path,'r')

for line in smile_file:
    vals=line.split(',')
    onesmile = (vals[0], vals[-1])
    smiles.append(onesmile)


fps, error_count, errors = getArrays(smiles)
print("Number of errors: ", error_count)
print("Compound IDs with errors: ", errors)

savePickle(fps, out_path, out_name)


