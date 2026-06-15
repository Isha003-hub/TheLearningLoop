let batchResults = [];

async function uploadCsv()
{
    
    let file =
        document.getElementById(
            "csvFile"
        ).files[0];

    let formData =
        new FormData();

    formData.append(
        "file",
        file
    );

    let response =
        await fetch(
            "http://127.0.0.1:8000/predict-csv",
            {
                method: "POST",
                body: formData
            }
        );

    let data =
        await response.json();
        batchResults =
        data.results;

    let html = "";

    data.results.forEach(item => {

        html += `
            <tr>
                <td>${item.smiles}</td>
                <td>${item.prediction}</td>
                <td>${item.confidence}%</td>
            </tr>
        `;
    });

    document.getElementById(
        "csvResults"
    ).innerHTML = html;
    
    document.getElementById(
        "batchResultsSection"
    ).style.display = "block";

    document.getElementById(
        "downloadCsvBtn"
    ).style.display = "inline-block";
}

function downloadResultsCsv()
{
    let csvContent =
        "SMILES,Prediction,Confidence\n";

    batchResults.forEach(row => {

        csvContent +=
            `${row.smiles},${row.prediction},${row.confidence}\n`;

    });

    let blob =
        new Blob(
            [csvContent],
            {
                type: "text/csv"
            }
        );

    let url =
        window.URL.createObjectURL(
            blob
        );

    let a =
        document.createElement(
            "a"
        );

    a.href = url;

    a.download =
        "prediction_results.csv";

    a.click();

    window.URL.revokeObjectURL(
        url
    );
}