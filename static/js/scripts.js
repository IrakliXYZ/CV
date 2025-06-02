// static/js/scripts.js
document.addEventListener('DOMContentLoaded', function() {
    // File upload handling
    const fileInput = document.getElementById('cv_file');
    const fileName = document.getElementById('file-name');
    const form = document.getElementById('cv-form');
    const submitBtn = document.getElementById('submit-btn');
    const loadingSpinner = document.getElementById('loading-spinner');
    const btnText = document.getElementById('btn-text');
    
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const file = this.files[0];
                fileName.textContent = file.name;
                
                // File size validation
                if (file.size > 5 * 1024 * 1024) { // 5MB
                    alert('File size exceeds 5MB limit.');
                    this.value = '';
                    fileName.textContent = 'No file selected';
                }
            } else {
                fileName.textContent = 'No file selected';
            }
        });
    }
    
    if (form) {
        form.addEventListener('submit', function() {
            // Show loading state
            if (submitBtn) {
                submitBtn.disabled = true;
                if (loadingSpinner) loadingSpinner.classList.remove('d-none');
                if (btnText) btnText.textContent = 'Processing...';
            }
        });
    }
});