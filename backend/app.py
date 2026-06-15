from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.predictor import predict_smiles
from backend.database import get_predictions

from fastapi import UploadFile, File
import shutil

from rdkit import Chem
from rdkit.Chem import Draw
from fastapi.staticfiles import StaticFiles

from backend.database import get_dashboard_stats

from backend.database import get_top_compounds

import csv


app = FastAPI()

app.mount(
    "/molecule_images",
    StaticFiles(
        directory="molecule_images"
    ),
    name="molecule_images"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to LearningLoop API"
    }


@app.get("/predict")
def predict(smiles: str):

    result = predict_smiles(smiles)

    return result

@app.get("/history")
def history(
    page: int = 1,
    limit: int = 10
):

    return get_predictions(
        page,
        limit
    )

@app.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...)
):

    file_path = f"uploads/{file.filename}"

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    return {
        "message": "CSV uploaded successfully",
        "filename": file.filename
    }

@app.post("/read-csv")
async def read_csv(
    file: UploadFile = File(...)
):

    rows = []

    contents = await file.read()

    decoded = contents.decode(
        "utf-8"
    ).splitlines()

    reader = csv.DictReader(
        decoded
    )

    for row in reader:

        rows.append(
            row["SMILES"]
        )

    return {
        "total_compounds": len(rows),
        "compounds": rows
    }

@app.post("/predict-csv")
async def predict_csv(
    file: UploadFile = File(...)
):

    results = []

    contents = await file.read()

    decoded = contents.decode(
        "utf-8"
    ).splitlines()

    reader = csv.DictReader(
        decoded
    )

    for row in reader:

        smiles = row["SMILES"]

        prediction = predict_smiles(
            smiles
        )

        results.append({
            "smiles": smiles,
            "prediction": prediction.get(
                "prediction"
            ),
            "confidence": prediction.get(
                "confidence"
            )
        })

    return {
        "total_compounds": len(results),
        "results": results
    }

@app.get("/molecule-image")
def molecule_image(smiles: str):

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:

        return {
            "error": "Invalid SMILES"
        }

    image_path = (
        f"molecule_images/{smiles}.png"
    )

    img = Draw.MolToImage(
        mol,
        size=(400, 300)
    )

    img.save(
        image_path
    )

    return {
        "message": "Image generated",
        "image": image_path
    }

@app.get("/dashboard-stats")
def dashboard_stats():

    return get_dashboard_stats()

@app.get("/top-compounds")
def top_compounds():

    return get_top_compounds()