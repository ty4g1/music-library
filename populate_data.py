import os
import json
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
import librosa
import numpy as np
import pandas as pd
import joblib
from math import floor
import traceback
import warnings
warnings.filterwarnings('ignore')

def extract_audio_features_from_signal(y, sr):
    """
    Extract features from an audio signal that's already loaded
    Returns a dictionary with named features in the correct order
    """
    try:
        # Define feature names in the exact order they were during training
        feature_names = [
            'chroma_stft_mean', 'rms_mean', 'spectral_centroid_mean',
            'spectral_bandwidth_mean', 'rolloff_mean', 'zero_crossing_rate_mean',
            'harmony_mean', 'perceptr_mean', 'tempo',
            'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean',
            'mfcc5_mean', 'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean',
            'mfcc9_mean', 'mfcc10_mean', 'mfcc11_mean', 'mfcc12_mean',
            'mfcc13_mean', 'mfcc14_mean', 'mfcc15_mean', 'mfcc16_mean',
            'mfcc17_mean', 'mfcc18_mean', 'mfcc19_mean', 'mfcc20_mean'
        ]
        
        features = {}
        
        # Compute STFT
        stft = librosa.stft(y)
        
        # Extract features in the same order
        features['chroma_stft_mean'] = np.mean(librosa.feature.chroma_stft(S=np.abs(stft)))
        features['rms_mean'] = np.mean(librosa.feature.rms(y=y))
        features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features['harmony_mean'] = np.mean(y_harmonic)
        features['perceptr_mean'] = np.mean(y_percussive)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
        
        # Create DataFrame with features in correct order
        features_df = pd.DataFrame([features])[feature_names]
        
        return features_df
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def predict_mood_segments(audio_path, model_path, scaler_path, segment_length=30, min_length=15):
    """
    Predict mood for each segment of the audio file
    """
    try:
        # Get features for all segments
        segments_features = process_audio_segments(audio_path, segment_length, min_length)
        
        if not segments_features:
            return None
            
        # Load scaler and model
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        
        # Process each segment
        predictions = []
        for features_df in segments_features:
            # Features are already in a DataFrame with correct column names
            # Normalize features
            features_normalized = pd.DataFrame(
                scaler.transform(features_df),
                columns=features_df.columns
            )
            # Predict
            pred = model.predict(features_normalized)[0]
            predictions.append(pred)
        
        return predictions
        
    except Exception as e:
        print(f"Error predicting moods: {str(e)}")
        return None

def process_audio_segments(audio_path, segment_length=30, min_length=15):
    """
    Split audio into segments and extract features from each valid segment
    """
    try:
        # Load the full audio file
        print(audio_path)
        y, sr = librosa.load(audio_path)

        # Calculate total duration in seconds
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate number of complete segments
        n_segments = floor(duration / segment_length)
        
        # Initialize list to store features for each segment
        segments_features = []
        
        # Process complete segments
        for i in range(n_segments):
            start_sample = i * segment_length * sr
            end_sample = (i + 1) * segment_length * sr
            segment = y[start_sample:end_sample]
            
            # Extract features for this segment
            features_df = extract_audio_features_from_signal(segment, sr)
            if features_df is not None:
                segments_features.append(features_df)
        
        # Process the last segment if it's long enough
        remaining_samples = len(y) - (n_segments * segment_length * sr)
        if remaining_samples > min_length * sr:
            last_segment = y[n_segments * segment_length * sr:]
            features_df = extract_audio_features_from_signal(last_segment, sr)
            if features_df is not None:
                segments_features.append(features_df)
        
        return segments_features
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def create_github_url(filename: str, github_username: str) -> str:
    """Create GitHub raw content URL for the audio file"""
    # URL encode the filename to handle spaces and special characters
    safe_filename = filename.replace(" ", "%20")
    return f"https://raw.githubusercontent.com/{github_username}/music-library/main/audio/{safe_filename}"

def get_predominant_mood(segments: List[str]) -> str:
    """Determine the most frequent mood from the segments"""
    if not segments:
        return "unknown"
    return Counter(segments).most_common(1)[0][0]

def process_audio_files(
    audio_folder: str,
    model_path: str,
    scaler_path: str,
    github_username: str,
    output_json: str
) -> None:
    """
    Process all audio files and create a JSON with mood analysis results
    """
    # Initialize results dictionary
    results: Dict[str, Dict] = {}
    
    # Use Path object for reliable path handling
    audio_path = Path(audio_folder)
    
    # Process each audio file
    count = 0
    for file_path in audio_path.glob("*.wav"):
        try:
            print(f"\nProcessing: {file_path}")
            
            # Get filename without extension for track_id
            track_id = file_path.stem
            
            print(f"Extracting features for: {track_id}")
            segments = predict_mood_segments(str(file_path), model_path, scaler_path)
            
            if segments is None:
                print(f"Error: No segments returned for {file_path}")
                continue
            
            # Create result entry
            results[track_id] = {
                "title": track_id,
                "file_url": create_github_url(file_path.name, github_username),
                "segments": segments,
                "predominant_mood": get_predominant_mood(segments)
            }
            
            print(f"Successfully processed: {track_id}")
            print(f"Found {len(segments)} segments with moods: {segments}")
            count += 1
            print(count, "/", 516)
        except Exception as e:
            print(f"\nError processing {file_path}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            print(traceback.format_exc())
            continue
    
    # Save results to JSON file
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully saved results to {output_json}")
        print(f"Processed {len(results)} files")
    except Exception as e:
        print(f"\nError saving JSON file:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

if __name__ == "__main__":
    # Configuration
    AUDIO_FOLDER = "audio"
    MODEL_PATH = "emotion_classifier.joblib"
    SCALER_PATH = "scaler.pkl"
    GITHUB_USERNAME = "ty4g1"  # Replace with actual username
    OUTPUT_JSON = "mood_analysis.json"

    print("Starting audio processing...")
    print(f"Looking for WAV files in: {AUDIO_FOLDER}")
    print(f"Using model: {MODEL_PATH}")
    print(f"Using scaler: {SCALER_PATH}")
    
    # Process all files
    process_audio_files(
        audio_folder=AUDIO_FOLDER,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        github_username=GITHUB_USERNAME,
        output_json=OUTPUT_JSON
    )