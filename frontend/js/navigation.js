function showAnalyze()
{
    document.getElementById(
        "analyzeTab"
    ).classList.add("active");

    document.getElementById(
        "historyTab"
    ).classList.remove("active");

    document.getElementById(
        "dashboardTab"
    ).classList.remove("active");

    document.getElementById(
        "analyzeSection"
    ).style.display = "block";

    document.getElementById(
        "dynamicContent"
    ).innerHTML = "";
}

async function loadHistory()
{
    document.getElementById(
        "analyzeSection"
    ).style.display = "none";

    document.getElementById(
        "historyTab"
    ).classList.add("active");

    document.getElementById(
        "dashboardTab"
    ).classList.remove("active");

    document.getElementById(
        "analyzeTab"
    ).classList.remove("active");

    const response =
        await fetch(
            "views/history.html"
        );

    const html =
        await response.text();

    document.getElementById(
        "dynamicContent"
    ).innerHTML = html;

    loadHistoryData();
}

async function loadDashboard()
{
    document.getElementById(
        "analyzeSection"
    ).style.display = "none";

    document.getElementById(
        "dashboardTab"
    ).classList.add("active");

    document.getElementById(
        "historyTab"
    ).classList.remove("active");

    document.getElementById(
        "analyzeTab"
    ).classList.remove("active");

    const response =
        await fetch(
            "views/dashboard.html"
        );

    const html =
        await response.text();

    document.getElementById(
        "dynamicContent"
    ).innerHTML = html;

    loadDashboardStats();

    loadTopCompounds();
}