async function loadDashboardStats()
{

    let response =
    await fetch(
        "/dashboard-stats"
    );
 /*   let response =
        await fetch(
            "http://127.0.0.1:8000/dashboard-stats"
        ); 

*/

    let data =
        await response.json();

    document.getElementById(
        "totalPredictions"
    ).innerText =
        data.total_predictions;

    document.getElementById(
        "activePredictions"
    ).innerText =
        data.active_predictions;

    document.getElementById(
        "inactivePredictions"
    ).innerText =
        data.inactive_predictions;

    createChart(
        data.active_predictions,
        data.inactive_predictions
    );
}

function createChart(
    active,
    inactive
)
{
    new Chart(
        document.getElementById(
            "predictionChart"
        ),
        {
            type: "pie",

            data: {
                labels: [
                    "Active",
                    "Inactive"
                ],

                datasets: [{
                    data: [
                        active,
                        inactive
                    ]
                }]
            }
        }
    );
}

async function loadTopCompounds()
{

    let response =
    await fetch(
        "/top-compounds"
    );
    
/*    let response =
        await fetch(
            "http://127.0.0.1:8000/top-compounds"
        );
*/
    let data =
        await response.json();

    let labels = [];
    let values = [];

    data.forEach(item => {

        labels.push(
            item.smiles
        );

        values.push(
            item.count
        );

    });

    new Chart(
        document.getElementById(
            "topCompoundsChart"
        ),
        {
            type: "bar",

            data: {
                labels: labels,

                datasets: [{
                    label: "Search Count",
                    data: values
                }]
            }
        }
    );
}