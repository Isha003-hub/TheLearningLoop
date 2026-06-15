from pathlib import Path
import torch

from backend.model_architecture import TheLearningLoopModel
from backend.database import save_prediction

MODEL_PATH = Path("model/best_hybrid_model.pth")


def load_model():

    model = TheLearningLoopModel()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print("Model loaded successfully!")

    return model



from torch_geometric.loader import DataLoader
from backend.preprocessing import smiles_to_hybrid_graph
import torch.nn.functional as F


def predict_smiles(smiles):

    data = smiles_to_hybrid_graph(smiles)

    if data is None:
        return {
            "error": "Invalid SMILES"
        }

    data_loader = DataLoader(
        [data],
        batch_size=1
    )

    model = load_model()

    with torch.no_grad():

        batch = next(iter(data_loader))

        outputs = model(batch)

        probs = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probs,
            dim=1
        )

    prediction = (
        "Active"
        if predicted.item() == 1
        else "Inactive"
    )

    confidence_value = round(
        confidence.item() * 100,
        2
    )

    save_prediction(
        smiles,
        prediction,
        confidence_value
    )

    return {
        "prediction": prediction,
        "confidence": confidence_value
    }