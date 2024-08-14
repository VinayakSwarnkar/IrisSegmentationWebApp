document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const message = document.getElementById('message');
    const uploadedImage = document.getElementById('uploadedImage');
    const segmentedImage = document.getElementById('segmentedImage');
    const downloadButton = document.getElementById('downloadButton');
    const processingIndicator = document.getElementById('processing');

    // Handle file selection via click or drag and drop
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', handleFileSelect);

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('highlight');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('highlight');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('highlight');
        handleFileSelect(e);
    });

    function handleFileSelect(event) {
        const file = event.target.files ? event.target.files[0] : event.dataTransfer.files[0];

        if (file) {
            const formData = new FormData();
            formData.append('file', file);

            // Show processing indicator
            processingIndicator.style.display = 'block';

            fetch('https://telemedc.pythonanywhere.com/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide processing indicator
                processingIndicator.style.display = 'none';

                if (data.original_image && data.segmented_image && data.original_resolution_segmented_image) {
                    uploadedImage.src = data.original_image;
                    segmentedImage.src = data.segmented_image;
                    uploadedImage.style.display = 'block';
                    segmentedImage.style.display = 'block';
                    message.textContent = 'Segmentation complete!';
                    downloadButton.style.display = 'block';

                    // Set the URL for downloading the original resolution segmented image
                    downloadButton.setAttribute('data-original-resolution-url', data.original_resolution_segmented_image);
                } else {
                    message.textContent = 'Error processing image.';
                }
            })
            .catch(error => {
                // Hide processing indicator
                processingIndicator.style.display = 'none';
                console.error('Error:', error);
                message.textContent = 'Error: ' + error.message;
            });
        }
    }

    window.downloadSegmentedImage = function() {
        const link = document.createElement('a');
        link.href = downloadButton.getAttribute('data-original-resolution-url');
        link.download = 'segmented_image_original_resolution.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };
});
