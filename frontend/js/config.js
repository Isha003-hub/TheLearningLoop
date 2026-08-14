const API_BASE_URL =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
        ? "http://127.0.0.1:8000"
        : "https://gnn-learningloop-docker-hufpg3d4hrf6abcn.canadacentral-01.azurewebsites.net";