document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const resultContent = document.getElementById('resultContent');
    const errorAlert = document.getElementById('errorAlert');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide previous results and errors
        result.style.display = 'none';
        errorAlert.style.display = 'none';
        
        // Show loading spinner
        loading.style.display = 'block';
        
        // Collect form data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });
        
        try {
            // Send prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            
            const predictionResult = await response.json();
            
            // Display results
            resultContent.innerHTML = `
                <div class="alert ${predictionResult.prediction === '>50K' ? 'alert-success' : 'alert-info'}">
                    <h4 class="alert-heading">Prediction Result</h4>
                    <p>Income: ${predictionResult.prediction}</p>
                    <p>Confidence: ${(predictionResult.probability * 100).toFixed(2)}%</p>
                </div>
                
                <div class="card mt-4">
                    <div class="card-header">
                        <h5 class="mb-0">Feature Importance</h5>
                    </div>
                    <div class="card-body">
                        <div class="feature-importance">
                            ${Object.entries(predictionResult.feature_importance)
                                .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
                                .map(([feature, importance]) => `
                                    <div class="feature-label">${feature}</div>
                                    <div class="feature-bar" style="width: ${Math.abs(importance * 100)}%; 
                                        background-color: ${importance > 0 ? '#28a745' : '#dc3545'};">
                                    </div>
                                `).join('')}
                        </div>
                    </div>
                </div>
            `;
            
            result.style.display = 'block';
        } catch (error) {
            errorAlert.textContent = 'An error occurred while making the prediction. Please try again.';
            errorAlert.style.display = 'block';
        } finally {
            loading.style.display = 'none';
        }
    });
}); 