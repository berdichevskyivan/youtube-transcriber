const ytdl = require('ytdl-core');
const fs = require('fs');
const path = require('path');
const deepspeech = require('deepspeech');
const ffmpeg = require('ffmpeg-static');
const cmd = require('node-cmd');

const modelPath = './deepspeech-0.8.0-models.pbmm';  // Path to the DeepSpeech model file
const scorerPath = './deepspeech-0.8.0-models.scorer';  // Path to the scorer file

const model = new deepspeech.Model(modelPath);
model.enableExternalScorer(scorerPath);

const downloadAndTranscribe = async (url) => {
    const videoId = new URL(url).searchParams.get('v');
    const audioOutput = path.resolve(__dirname, `${videoId}.wav`);
    const pcmAudioOutput = path.resolve(__dirname, `${videoId}.pcm`);
    const transcriptionOutput = path.resolve(__dirname, `${videoId}.txt`);

    console.log('Downloading audio...');
    const stream = ytdl(url, { filter: 'audioonly' });
    stream.pipe(fs.createWriteStream(audioOutput))
        .on('finish', () => {
            console.log('Audio downloaded. Converting and transcribing...');
            const command = `${ffmpeg} -i ${audioOutput} -ac 1 -ar 16000 -sample_fmt s16 ${pcmAudioOutput}`;
            cmd.run(command, (err, data, stderr) => {
                if (err || stderr) {
                    console.error('Error:', err || stderr);
                    return;
                }
                const buffer = fs.readFileSync(pcmAudioOutput);
                const result = model.stt(buffer);
                fs.writeFileSync(transcriptionOutput, result);
                console.log(`Transcription saved to ${transcriptionOutput}`);
            });
        });
};

downloadAndTranscribe('https://www.youtube.com/watch?v=VmPuh7wUEfY');  // Replace with actual YouTube video ID
