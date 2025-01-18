from pathlib import Path
import os
from pydub import AudioSegment


def convert_wav_to_mp3(wav_path: Path) -> bool:
    """
    Convert a WAV file to MP3 format.
    Returns True if conversion was successful, False otherwise.
    """
    try:
        print(f"Converting: {wav_path}")
        # Load WAV file
        audio = AudioSegment.from_wav(str(wav_path))
        
        # Create MP3 path with same name
        mp3_path = wav_path.with_suffix('.mp3')
        
        # Export as MP3
        audio.export(str(mp3_path), format='mp3')
        print(f"Successfully created: {mp3_path}")
        return True
        
    except Exception as e:
        print(f"Error converting {wav_path}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return False

def convert_all_files(audio_folder: str, delete_wavs: bool = False) -> None:
    """
    Convert all WAV files in the specified folder to MP3 format.
    
    Parameters:
    - audio_folder: Path to folder containing WAV files
    - delete_wavs: If True, delete original WAV files after successful conversion
    """
    audio_path = Path(audio_folder)
    
    if not audio_path.exists():
        print(f"Error: Folder {audio_folder} does not exist!")
        return
    
    # Find all WAV files
    wav_files = list(audio_path.glob("*.wav"))
    total_files = len(wav_files)
    print(total_files)
    if total_files == 0:
        print("No WAV files found in the specified folder.")
        return
    
    print(f"Found {total_files} WAV files to convert")
    
    # Convert each file
    successful_conversions = 0
    failed_conversions = 0
    
    for wav_file in wav_files:
        if convert_wav_to_mp3(wav_file):
            successful_conversions += 1
            if delete_wavs:
                try:
                    os.remove(wav_file)
                    print(f"Deleted original WAV file: {wav_file}")
                except Exception as e:
                    print(f"Warning: Could not delete {wav_file}: {str(e)}")
        else:
            failed_conversions += 1
    
    # Print summary
    print("\nConversion Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    if delete_wavs:
        print("Original WAV files were deleted after successful conversion")

if __name__ == "__main__":
    # Configuration
    AUDIO_FOLDER = "audio"
    DELETE_ORIGINALS = True  # Set to False if you want to keep WAV files
    
    print(f"Starting WAV to MP3 conversion in folder: {AUDIO_FOLDER}")
    print(f"Delete original WAV files: {DELETE_ORIGINALS}")
    
    # Convert all files
    convert_all_files(AUDIO_FOLDER, DELETE_ORIGINALS)