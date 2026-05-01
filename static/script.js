let chart;
let trainChart;

async function train() {
    const statusEl = document.getElementById("status");
    const btn = document.getElementById("btnTrain");
    statusEl.innerText = "Starting training...";
    btn.classList.add("loading");

    let episodes = document.getElementById("episodes").value;

    try {
        let res = await fetch("/train", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({episodes})
        });

        let data = await res.json();
        
        if (data.status === "error") {
            statusEl.innerText = "Error: " + data.message;
            btn.classList.remove("loading");
            return;
        }

        statusEl.innerText = "Training in progress (this may take a few minutes)...";
        pollTrainingStatus(btn, statusEl);

    } catch (error) {
        statusEl.innerText = "Network/JS Error: " + error.message;
        console.error(error);
        btn.classList.remove("loading");
    }
}

function pollTrainingStatus(btn, statusEl) {
    const interval = setInterval(async () => {
        try {
            let res = await fetch("/train_status");
            let data = await res.json();

            if (!data.is_training) {
                clearInterval(interval);
                btn.classList.remove("loading");
                
                if (data.error) {
                    statusEl.innerText = "Error: " + data.error;
                } else if (data.savings_trend) {
                    statusEl.innerText = data.message || "Training Completed!";
                    renderTrainChart(data.savings_trend);
                } else {
                    statusEl.innerText = data.message || "Training finished.";
                }
            } else {
                statusEl.innerText = "Training in progress...";
            }
        } catch (err) {
            console.error("Polling error:", err);
        }
    }, 2000);
}


async function test() {
    const statusEl = document.getElementById("status");
    const btn = document.getElementById("btnTest");
    statusEl.innerText = "Running Simulation...";
    btn.classList.add("loading");

    let steps = document.getElementById("steps").value;

    try {
        let res = await fetch("/test", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({steps})
        });

        let data = await res.json();
        
        if (data.error) {
            statusEl.innerText = "Error: " + data.error;
            btn.classList.remove("loading");
            return;
        }

        document.getElementById("cost").innerText = "₹ " + data.cost;
        document.getElementById("solar_charge").innerText = data.solar_charge;
        document.getElementById("savings").innerText = data.savings + "%";
        document.getElementById("sold_energy").innerText = data.sold_energy;

        
        renderChart(data.rewards, data.battery_levels, data.prices);

        statusEl.innerText = "Simulation completed!";
    } catch (error) {
        statusEl.innerText = "Error during simulation: " + error.message;
    } finally {
        btn.classList.remove("loading");
    }
}

function renderChart(rewards, battery, prices) {
    const ctx = document.getElementById("chart");

    if (chart) chart.destroy();

    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: rewards.map((_, i) => i),
            datasets: [
                {
                    label: "Electricity Cost (₹)",
                    data: rewards.map(r => -r),
                    borderColor: "#ff0055", // neon pink
                    backgroundColor: "rgba(255, 0, 85, 0.1)",
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: "#ff0055",
                    pointBorderColor: "#fff",
                    pointHoverBackgroundColor: "#fff",
                    pointHoverBorderColor: "#ff0055",
                    pointRadius: 0,
                    pointHoverRadius: 6
                },
                {
                    label: "Battery Charge Level (%)",
                    data: battery,
                    borderColor: "#39ff14", 
                    borderWidth: 3,
                    tension: 0.4,
                    pointBackgroundColor: "#39ff14",
                    pointRadius: 0,
                    pointHoverRadius: 6
                },
                {
                    label: "Grid Market Price (₹)",
                    data: prices,
                    borderColor: "#00f0ff", 
                    borderWidth: 3,
                    tension: 0.4,
                    borderDash: [5, 5],
                    pointBackgroundColor: "#00f0ff",
                    pointRadius: 0,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: { color: '#e2e8f0' }
                },
                tooltip: {
                    backgroundColor: 'rgba(11, 15, 25, 0.9)',
                    titleColor: '#00f0ff',
                    bodyColor: '#e2e8f0',
                    borderColor: 'rgba(0, 240, 255, 0.3)',
                    borderWidth: 1,
                    padding: 10
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Time Steps (Hours)', color: '#94a3b8' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Metric Value (₹ / %)', color: '#94a3b8' }
                }
            }
        }
    });
}

function renderTrainChart(savingsTrend) {
    const ctx = document.getElementById("trainChart");

    if (trainChart) trainChart.destroy();

    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    trainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: savingsTrend.map((_, i) => `Ep ${i+1}`),
            datasets: [
                {
                    label: "Cost Savings (%)",
                    data: savingsTrend,
                    borderColor: "#39ff14", // neon green
                    backgroundColor: "rgba(57, 255, 20, 0.1)",
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: "#39ff14",
                    pointRadius: 3,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: { color: '#e2e8f0' }
                },
                tooltip: {
                    backgroundColor: 'rgba(11, 15, 25, 0.9)',
                    titleColor: '#00f0ff',
                    bodyColor: '#e2e8f0',
                    borderColor: 'rgba(0, 240, 255, 0.3)',
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Training Episodes', color: '#94a3b8' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Savings (%)', color: '#94a3b8' }
                }
            }
        }
    });
}

function logout() {
    window.location.href = "/logout";
}