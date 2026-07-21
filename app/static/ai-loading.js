"use strict";

(() => {
    const loaders = document.querySelectorAll("[data-ai-analysis-loader]");
    if (!loaders.length) return;

    const storage = {
        get(key) {
            try {
                return window.sessionStorage.getItem(key);
            } catch (_error) {
                return null;
            }
        },
        set(key, value) {
            try {
                window.sessionStorage.setItem(key, value);
            } catch (_error) {
                // Der Status-Poll funktioniert auch ohne Session Storage.
            }
        },
        remove(key) {
            try {
                window.sessionStorage.removeItem(key);
            } catch (_error) {
                // Der Status-Poll funktioniert auch ohne Session Storage.
            }
        }
    };

    function stateFromStatus(status) {
        if (status?.state) return String(status.state);
        if (!status?.complete) return "pending";
        return status?.available ? "ready" : "error";
    }

    function startPolling(loader) {
        const statusUrl = loader.dataset.statusUrl;
        if (!statusUrl) return;

        const startedAt = Date.now();
        const recoveryKey = `ai-analysis-recovery:${loader.dataset.recoveryKey || statusUrl}`;
        const detail = loader.querySelector("[data-ai-loading-detail]");

        const poll = async () => {
            try {
                const response = await fetch(statusUrl, {headers: {"Accept": "application/json"}});
                const status = response.ok ? await response.json() : null;
                const state = stateFromStatus(status);
                if (state === "ready" || state === "error") {
                    storage.remove(recoveryKey);
                    window.location.reload();
                    return;
                }
            } catch (_error) {
                // Netzwerkfehler beenden das automatische Nachladen nicht.
            }

            const elapsedMs = Date.now() - startedAt;
            if (elapsedMs >= 30000 && detail) {
                detail.textContent = "Die KI-Auswertung dauert etwas länger. Diese Seite lädt sie weiterhin automatisch nach.";
            }
            if (elapsedMs >= 60000 && storage.get(recoveryKey) !== "reloaded") {
                storage.set(recoveryKey, "reloaded");
                window.location.reload();
                return;
            }
            window.setTimeout(poll, elapsedMs < 30000 ? 1000 : 3000);
        };

        window.setTimeout(poll, 700);
    }

    loaders.forEach(startPolling);
})();
