document.addEventListener('DOMContentLoaded', function() {
    // Fetch services status on page load
    fetchServicesStatus();

    // Add event listener for refresh button
    document.getElementById('refreshStatus').addEventListener('click', fetchServicesStatus);

    // Add event listener for inference form
    document.getElementById('inferenceForm').addEventListener('submit', function(event) {
        event.preventDefault();
        runInference();
    });

    // Add event listener for random features button
    document.getElementById('randomFeatures').addEventListener('click', function() {
        // Generate random values between 0 and 1
        document.getElementById('feature1').value = Math.random().toFixed(2);
        document.getElementById('feature2').value = Math.random().toFixed(2);
        document.getElementById('feature3').value = Math.random().toFixed(2);
        document.getElementById('feature4').value = Math.random().toFixed(2);
    });
});

/**
 * Fetch the status of all microservices
 */
function fetchServicesStatus() {
    // Show loading spinner
    document.getElementById('servicesStatus').innerHTML = `
        <div class="col-12 text-center">
            <div class="spinner-border text-info" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Loading services status...</p>
        </div>
    `;

    fetch('/api/services')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            displayServicesStatus(data);
            updateArchitectureDiagram(data);
        })
        .catch(error => {
            document.getElementById('servicesStatus').innerHTML = `
                <div class="col-12">
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Error fetching services status: ${error.message}
                    </div>
                </div>
            `;
        });
}

/**
 * Display the status of all microservices
 */
function displayServicesStatus(services) {
    const servicesStatusContainer = document.getElementById('servicesStatus');
    servicesStatusContainer.innerHTML = '';

    // Create a card for each service
    for (const [serviceName, serviceData] of Object.entries(services)) {
        const isHealthy = serviceData.health.status === 'healthy';
        const statusClass = isHealthy ? 'success' : 'danger';
        const statusIcon = isHealthy ? 'check-circle' : 'times-circle';
        
        const serviceCard = document.createElement('div');
        serviceCard.className = 'col-md-6 col-lg-3 mb-3';
        serviceCard.innerHTML = `
            <div class="card h-100">
                <div class="card-header bg-${statusClass} bg-opacity-25 d-flex justify-content-between align-items-center">
                    <h6 class="mb-0">${formatServiceName(serviceName)}</h6>
                    <span class="badge bg-${statusClass}">
                        <i class="fas fa-${statusIcon} me-1"></i>
                        ${isHealthy ? 'Healthy' : 'Unhealthy'}
                    </span>
                </div>
                <div class="card-body">
                    <p class="card-text small">${serviceData.info.description}</p>
                    <p class="card-text small mb-0">
                        <strong>URL:</strong> ${serviceData.info.url}
                    </p>
                </div>
            </div>
        `;
        
        servicesStatusContainer.appendChild(serviceCard);
    }
}

/**
 * Update the architecture diagram based on service status
 */
function updateArchitectureDiagram(services) {
    for (const [serviceName, serviceData] of Object.entries(services)) {
        const isHealthy = serviceData.health.status === 'healthy';
        const diagramElement = document.getElementById(`diagram-${serviceName}`);
        
        if (diagramElement) {
            const rect = diagramElement.querySelector('rect');
            if (rect) {
                rect.setAttribute('stroke', isHealthy ? 'var(--bs-success)' : 'var(--bs-danger)');
            }
        }
    }
}

/**
 * Format service name for display
 */
function formatServiceName(name) {
    return name.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

/**
 * Run inference with the current form values
 */
function runInference() {
    // Show loading in results area
    document.getElementById('inferenceResults').innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-info" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Running inference...</p>
        </div>
    `;

    // Get feature values from form
    const features = [
        parseFloat(document.getElementById('feature1').value),
        parseFloat(document.getElementById('feature2').value),
        parseFloat(document.getElementById('feature3').value),
        parseFloat(document.getElementById('feature4').value)
    ];

    // Send request to test inference endpoint
    fetch('/api/test-inference', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(features),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        displayInferenceResults(data);
    })
    .catch(error => {
        document.getElementById('inferenceResults').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                Error running inference: ${error.message}
            </div>
        `;
    });
}

/**
 * Display inference results
 */
function displayInferenceResults(data) {
    // Format the prediction value
    let prediction = 'N/A';
    if (data.prediction && Array.isArray(data.prediction)) {
        prediction = data.prediction.map(p => p.toFixed(4)).join(', ');
    }

    // Create HTML for the results
    let resultsHtml = `
        <div class="mb-3">
            <h5 class="border-bottom pb-2">Prediction Result</h5>
            <div class="mb-2">
                <strong>Prediction:</strong> 
                <span class="badge bg-primary fs-6">${prediction}</span>
            </div>
    `;

    // Show confidence if available
    if (data.postprocessing_info && data.postprocessing_info.confidence !== null) {
        resultsHtml += `
            <div class="mb-2">
                <strong>Confidence:</strong> 
                <span class="badge bg-info">${(data.postprocessing_info.confidence * 100).toFixed(2)}%</span>
            </div>
        `;
    }

    // Add timestamp if available
    if (data.timestamp) {
        const timestamp = new Date(data.timestamp).toLocaleString();
        resultsHtml += `
            <div class="mb-2">
                <strong>Timestamp:</strong> ${timestamp}
            </div>
        `;
    }

    // Add preprocessing information if available
    if (data.preprocessing_info) {
        resultsHtml += `
            <h5 class="border-bottom pb-2 mt-4">Preprocessing Details</h5>
            <div class="small">
                <div><strong>Mean:</strong> ${data.preprocessing_info.mean.toFixed(4)}</div>
                <div><strong>Std:</strong> ${data.preprocessing_info.std.toFixed(4)}</div>
                <div><strong>Missing Values Replaced:</strong> ${data.preprocessing_info.replaced_missing ? 'Yes' : 'No'}</div>
            </div>
        `;
    }

    // Add postprocessing information if available
    if (data.postprocessing_info) {
        resultsHtml += `
            <h5 class="border-bottom pb-2 mt-4">Postprocessing Details</h5>
            <div class="small">
                <div><strong>Modified:</strong> ${data.postprocessing_info.modified ? 'Yes' : 'No'}</div>
                <div><strong>Original Range:</strong> Min: ${data.postprocessing_info.original_range.min.toFixed(4)}, Max: ${data.postprocessing_info.original_range.max.toFixed(4)}</div>
            </div>
        `;
    }

    // Add raw JSON data collapsible section
    resultsHtml += `
        <div class="mt-4">
            <p class="mb-2">
                <button class="btn btn-sm btn-outline-secondary" type="button" data-bs-toggle="collapse" data-bs-target="#rawResponseData">
                    Show Raw Response Data
                </button>
            </p>
            <div class="collapse" id="rawResponseData">
                <div class="card card-body bg-dark">
                    <pre class="mb-0 text-light"><code>${JSON.stringify(data, null, 2)}</code></pre>
                </div>
            </div>
        </div>
    `;

    // Update the results container
    document.getElementById('inferenceResults').innerHTML = resultsHtml;
}
