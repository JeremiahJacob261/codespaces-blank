const express = require('express');
const path = require('path');

// Create an Express application
const app = express();

// Define the port for the server
const PORT = 3000;

// Serve files from the "images" folder
app.use('/images', express.static(path.join(__dirname, 'images')));

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}/images`);
});

// OCR pipeline
(async () => {
    try {
        // Dynamically import the pipeline from the @xenova/transformers module
        const { env } = await import('@xenova/transformers');

        env.localModelPath = 'models'; // Set the local model path

        // Disable the loading of remote models from the Hugging Face Hub:
        env.allowRemoteModels = false;

        env.backends.onnx.wasm.wasmPaths = 'wasm'
        const { pipeline } = await import('@xenova/transformers');

        // Create image-to-text pipeline
        const captioner = await pipeline('image-to-text', 'Xenova/trocr-base-handwritten');

        // Use a local image served by the server
        const image = 'http://localhost:3000/images/photo3.jpg';
        const output = await captioner(image);

        console.log(output); // [{ generated_text: 'Mr. Brown commented icily.' }]
    } catch (error) {
        console.error('Error:', error);
    }
})();
