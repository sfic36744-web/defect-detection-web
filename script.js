// File input handling
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const uploadForm = document.getElementById('uploadForm');

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                fileName.textContent = this.files[0].name;
                
                // Preview image if needed
                const reader = new FileReader();
                reader.onload = function(e) {
                    // You can add image preview here if needed
                };
                reader.readAsDataURL(this.files[0]);
            } else {
                fileName.textContent = 'No file chosen';
            }
        });
    }

    // Form submission with loading state
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const file = fileInput.files[0];
            if (!file) {
                e.preventDefault();
                alert('Please select an image file first.');
                return;
            }
            
            if (analyzeBtn) {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<div class="loading"></div> Analyzing...';
            }
        });
    }

    // API example usage for developers
    window.analyzeWithAPI = async function(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                console.log('Analysis result:', result.data);
                return result.data;
            } else {
                console.error('Analysis failed:', result.error);
                throw new Error(result.error);
            }
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    };

    // Example of how to use the API programmatically
    window.uploadAndAnalyze = function() {
        const fileInput = document.getElementById('fileInput');
        if (fileInput.files.length === 0) {
            alert('Please select a file first.');
            return;
        }

        analyzeWithAPI(fileInput.files[0])
            .then(result => {
                console.log('Analysis completed:', result);
                // You can display results in a custom way here
                alert(`Analysis completed: ${result.results.final_type}`);
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Analysis failed: ' + error.message);
            });
    };
});

// Print functionality
function printReport() {
    window.print();
}

// Image error handling
function handleImageError(img) {
    img.style.display = 'none';
    const placeholder = img.nextElementSibling;
    if (placeholder && placeholder.classList.contains('image-placeholder')) {
        placeholder.style.display = 'flex';
    }
}

// Add smooth loading for images
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('img[src]');
    images.forEach(img => {
        // Add loading attribute for better performance
        img.setAttribute('loading', 'lazy');
        
        // Handle image load errors
        img.addEventListener('error', function() {
            handleImageError(this);
        });
        
        // Add fade-in effect when image loads
        img.addEventListener('load', function() {
            this.style.opacity = '0';
            this.style.transition = 'opacity 0.5s ease';
            setTimeout(() => {
                this.style.opacity = '1';
            }, 100);
        });
    });
});