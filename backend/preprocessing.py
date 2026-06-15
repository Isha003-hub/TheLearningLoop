import torch

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from torch_geometric.data import Data


morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


def smiles_to_hybrid_graph(smiles):

    try:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        xs = [
            [atom.GetAtomicNum()]
            for atom in mol.GetAtoms()
        ]

        x = torch.tensor(
            xs,
            dtype=torch.float
        )

        edge_indices = []

        for bond in mol.GetBonds():

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_indices.extend([
                [i, j],
                [j, i]
            ])

        edge_index = torch.tensor(
            edge_indices,
            dtype=torch.long
        ).t().contiguous()

        fp_numpy = morgan_gen.GetFingerprintAsNumPy(
            mol
        )

        fp_tensor = torch.tensor(
            fp_numpy,
            dtype=torch.float
        ).view(1, -1)

        data = Data(
            x=x,
            edge_index=edge_index,
            fp=fp_tensor
        )

        return data

    except Exception:
        return None