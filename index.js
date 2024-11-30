const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const Jimp = require('jimp'); // For image processing

// Load the model
async function loadModel() {
    const model = await tf.loadLayersModel('model.json');
    return model;
}

// Preprocess the image
async function preprocessImage(imagePath) {
    const image = await Jimp.read(imagePath);
    image.resize(128, 32); // Resize to match model input size (adjust as necessary)
    const imageData = new Uint8Array(image.bitmap.data);
    
    // Normalize pixel values to [0, 1]
    const tensor = tf.tensor3d(imageData, [image.bitmap.height, image.bitmap.width, 4]);
    const normalizedTensor = tensor.slice([0, 0, 0], [image.bitmap.height, image.bitmap.width, 3]).div(255);
    
    return normalizedTensor.expandDims(0); // Add batch dimension
}

// Recognize text from the image
async function recognizeText(model, imagePath) {
    const processedImage = await preprocessImage(imagePath);
    const predictions = model.predict(processedImage);
    
    // Assuming the output is character probabilities; decode them accordingly.
    const predictedIndices = predictions.argMax(-1).dataSync();
    
    // Convert indices to characters (assuming a fixed alphabet)
    const alphabet = 'abcdefghijklmnopqrstuvwxyz123456789'; // Adjust based on your model's training
    let recognizedText = '';
    
    predictedIndices.forEach(index => {
        recognizedText += alphabet[index];
    });

    return recognizedText;
}

// Main function to run the recognition
(async () => {
    try {
        const model = await loadModel();
        const imagePath = 'three.jpg'; // Replace with your image path
        const text = await recognizeText(model, imagePath);
        console.log('Recognized Text:', text);
    } catch (error) {
        console.error('Error:', error);
    }
})();