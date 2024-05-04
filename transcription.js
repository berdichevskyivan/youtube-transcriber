const fs = require('fs');
const deepspeech = require('deepspeech');
const path = require('path');

// Load the DeepSpeech model
const modelPath = './deepspeech-0.8.0-models.pbmm';
const scorerPath = './deepspeech-0.8.0-models.scorer';
const model = new deepspeech.Model(modelPath);
model.enableExternalScorer(scorerPath);

// Adjust transcription parameters for optimization
model.setBeamWidth(500); // Set beam width parameter
model.setScorerAlphaBeta(0.75, 1.85); // Set LM alpha and beta parameters

// Create a DeepSpeech stream
const stream = model.createStream();

// Specify the path to the PCM audio file
const pcmAudioFilePath = path.join(__dirname, 'VmPuh7wUEfY.pcm');

console.log('Reading PCM audio file...');
// Read PCM audio file
const audioBuffer = fs.readFileSync(pcmAudioFilePath);
console.log('PCM audio file read successfully.');

// Specify the number of bytes to print from the beginning of the audio buffer
const bytesToPrint = 100;

// Print the first few bytes of the audio buffer
console.log('First few bytes of the audio buffer:');
console.log(audioBuffer.slice(0, bytesToPrint));

console.log('Feeding audio content to the stream...');
// Feed audio content to the stream
stream.feedAudioContent(audioBuffer);
console.log('Audio content fed to the stream successfully.');

console.log('Finishing the stream and getting the transcription...');
// Finish the stream and get the transcription
const transcription = stream.finishStream();
console.log('Stream finished successfully.');

// Print or save the transcription
console.log('Transcription:', transcription);
