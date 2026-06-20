async function predict()
{
    let smiles =
        document.getElementById("smiles").value;

    let response =
        await fetch(
            "http://127.0.0.1:8000/predict?smiles="
            + encodeURIComponent(smiles)
        );

    let data =
        await response.json();

        await fetch(
            "http://127.0.0.1:8000/molecule-image?smiles="
            + encodeURIComponent(smiles)
        );
    document.querySelector('.result-section').style.display = 'flex';
    document.getElementById("result").innerHTML =
`
<div>

    <h4 class="text-center mb-4">
        Prediction Result
    </h4>
            
    <div class="result-block">
        <div class="item">
            <p class="text-xs text-muted mb-0">RESULT</p>
            <p class="mb-0"><strong>${data.prediction}</strong></p>
        </div>
        
        <div class="item">
            <p class="text-xs text-muted mb-0">CONFIDENCE</p>
            <p class="mb-0"><strong>${data.confidence}%</strong></p>
        </div>
    </div>

</div>
`;

    let image =
        document.getElementById(
            "moleculeImage"
        );

    image.src =
        "http://127.0.0.1:8000/molecule_images/"
        + encodeURIComponent(smiles)
        + ".png?t="
        + new Date().getTime();

    image.style.display =
        "block";
}
