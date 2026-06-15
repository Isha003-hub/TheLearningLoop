let currentPage = 1;
let totalPages = 1;

async function loadHistoryData(page = 1)
{
    let response =
        await fetch(
            `http://127.0.0.1:8000/history?page=${page}&limit=10`
        );

    let result =
        await response.json();

    let html = "";

    result.data.forEach(item => {

        html += `
            <tr>
                <td>${item.id}</td>
                <td>${item.smiles}</td>
                <td>${item.prediction}</td>
                <td>${item.confidence}%</td>
                <td>${item.created_at}</td>
            </tr>
        `;
    });

    document.getElementById(
        "historyTable"
    ).innerHTML = html;

    currentPage = result.page;
    totalPages = result.pages;

    const pageInfo =
        document.getElementById("pageInfo");

    if (pageInfo)
    {
        pageInfo.innerText =
            `Page ${result.page} of ${result.pages}`;
    }
}

function nextPage()
{
    if (currentPage < totalPages)
    {
        loadHistoryData(
            currentPage + 1
        );
    }
}

function previousPage()
{
    if (currentPage > 1)
    {
        loadHistoryData(
            currentPage - 1
        );
    }
}