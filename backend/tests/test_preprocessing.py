from preprocessing import smiles_to_hybrid_graph

data = smiles_to_hybrid_graph("CCO")

print(data)
print(data.x.shape)
print(data.edge_index.shape)
print(data.fp.shape)