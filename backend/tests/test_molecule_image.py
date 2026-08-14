from rdkit import Chem
from rdkit.Chem import Draw

smiles = "CCO"

mol = Chem.MolFromSmiles(smiles)

img = Draw.MolToImage(
    mol,
    size=(400, 300)
)

img.save(
    "ethanol.png"
)

print(
    "Image created successfully!"
)