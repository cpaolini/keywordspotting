import os
import time
import numpy as np
import librosa
from scipy.io.wavfile import write, read as wavread
import vart
import xir

def apply_preemphasis(signal, preemphasis):
    return np.append(signal[0], signal[1:] - preemphasis * signal[:-1])

def record_audio_arecord(filename, duration=1, sample_rate=44100):
    time.sleep(1)8d54233a496a9d3231df1cfd1075ee91bf920b61be4045db
    command = f"arecord -D plughw:1,0 -f S16_LE -c 1 -r {sample_rate} -d {duration} {filename}"
    result = os.system(command)
    if result != 0:
        print(f"Recording failed with exit code {result}")

def extract_features_from_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    preemphasis = 0.985
    sample_rate, audio = wavread(file_path)
    audio = audio.astype(np.float32)
    audio = apply_preemphasis(audio, preemphasis)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
    combined = np.vstack((mfcc, chroma, mel, contrast, tonnetz))
    return combined

def main():
    filename = "recording.wav"
    duration = 1
    sample_rate = 44100

    try:
        features = extract_features_from_audio("recording.wav")
        print("Features extracted successfully.")
        print("Features shape:", features.shape)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return

    try:
        graph = xir.Graph.deserialize("model.xmodel")
        subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
        dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_tensors = dpu_runner.get_input_tensors()
    output_tensors = dpu_runner.get_output_tensors()
    input_shape = input_tensors[0].dims

    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)
    features = np.float32(features)
    
    if features.shape != tuple(input_shape):
        print(f"Error: input shape mismatch. Expected: {tuple(input_shape)}, but got: {features.shape}")
        return

    try:
        input_data = {input_tensors[0].name: features}
        output_data = {output_tensors[0].name: np.empty(output_tensors[0].dims, dtype=np.float32)}
        job_id = dpu_runner.execute_async(input_data, output_data)
        dpu_runner.wait(job_id)
        
        predictions = output_data[output_tensors[0].name]
        predicted_class = np.argmax(predictions)
        confidence_score = predictions[0][predicted_class]
        labels = ['blinds', 'fan', 'light', 'music', 'tv']
        print(f"Predicted class: {labels[predicted_class]}")
        print(f"confidence score: {confidence_score:.2f}")
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
