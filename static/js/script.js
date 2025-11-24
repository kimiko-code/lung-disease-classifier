// Image preview
document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const preview = document.getElementById('imagePreview');

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } else {
        preview.src = '#';
        preview.style.display = 'none';
    }
});

// Prediction request
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const results = document.getElementById('results');
    const prediction = document.getElementById('prediction');
    const confidence = document.getElementById('confidence');
    const probabilitiesList = document.getElementById('probabilitiesList');

    loading.style.display = 'block';
    error.textContent = '';
    results.style.display = 'none';

    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files[0]) {
        error.textContent = 'Please select an image first.';
        loading.style.display = 'none';
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const res = await fetch('/predict', { method: 'POST', body: formData });

        if (!res.ok) {
            throw new Error(await res.text());
        }

        const data = await res.json();

        prediction.textContent = data.prediction;
        confidence.textContent = (data.confidence * 100).toFixed(2);

        probabilitiesList.innerHTML = '';
        for (const [label, prob] of Object.entries(data.class_probabilities)) {
            const li = document.createElement('li');
            li.textContent = `${label}: ${(prob * 100).toFixed(2)}%`;
            probabilitiesList.appendChild(li);
        }

        results.style.display = 'block';
    } catch (err) {
        error.textContent = err.message || 'An error occurred';
    } finally {
        loading.style.display = 'none';
    }
});
