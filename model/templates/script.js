function analyze() {
    let symptoms = document.getElementById("symptoms").value;
    let history = document.getElementById("history").value;

    fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            symptoms: symptoms,
            previous_conditions: history
        })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerHTML = `
            <b>Severity:</b> ${data.severity}
        `;
    })
    .catch(err => {
        console.error(err);
        document.getElementById("result").innerHTML = "Error: Could not reach backend.";
    });
}
