import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors
import numpy as np

class SM_Chem():
    'Generates morgan fingerprint with RDkit'
    def __init__(self, radius=2, bits=2048):# total length is (radius+1) * bits
        self.radius = radius
        self.bit = bits
               
    def get_binary(self,smiles):
        
        binary = np.zeros((2048*(self.radius+1)), int)
        mol = Chem.MolFromSmiles(smiles)
        mol_H = Chem.AddHs(mol)
        mol_bi_H = {}
        for r in range(self.radius+1):
            mol_fp_H = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_H, radius=self.radius, bitInfo=mol_bi_H, nBits = self.bit)
            mol_bi_H_QC = []
            for i in mol_fp_H.GetOnBits():
                idx = mol_bi_H[i][0][0]
                radius_list = []
                for j in range(len(mol_bi_H[i])):
                    atrom_radius = mol_bi_H[i][j][1]
                    radius_list.append(atrom_radius)
                #atom = mol_H.GetAtomWithIdx(idx)
                #symbol = atom.GetSymbol()
                #neighbor = [x.GetAtomicNum() for x in atom.GetNeighbors()]
                if r in radius_list:
                    mol_bi_H_QC.append(i)
            bits = mol_bi_H_QC
            for i in bits:
                binary[(2048*r)+i] = 1
        return binary
    
    def smiles_aug(self,smiles):
        try: #sometimes, RDkit can't handle some type of SMILES randomly.
            smiles_v = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),isomericSmiles=False, doRandom = True)
        except:
            smiles_v = smiles
        return smiles_v
            
    def smiles_valid(self, smile):
        try:
            return Chem.MolFromSmiles(smile) is not None
        except:
            return False