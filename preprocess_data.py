import json
from collections import defaultdict

def process_music_database():
    # Load original database
    with open('data/mood_analysis.json', 'r') as file:
        music_database = json.load(file)
    
    # Initialize processed structure
    processed_data = defaultdict(lambda: {
        "pure": [],
        "transitions": defaultdict(list)
    })
    
    # Process each song
    for song_title, song_data in music_database.items():
        segments = song_data["segments"]
        if not segments:  # Skip if no segments
            continue
            
        start_mood = segments[0]
        end_mood = segments[-1]
        
        # Create song entry with relevant data
        song_entry = {
            "title": song_title,
            "file_url": song_data["file_url"],
            "segments": segments,
            "predominant_mood": song_data["predominant_mood"]
        }
        
        # If start and end moods are same, it's a pure mood song
        if start_mood == end_mood:
            processed_data[start_mood]["pure"].append(song_entry)
        
        # Add to transitions regardless (includes pure mood songs too)
        transition_key = f"{start_mood}_to_{end_mood}"
        processed_data[start_mood]["transitions"][transition_key].append(song_entry)
    
    # Convert defaultdict to regular dict for JSON serialization
    final_data = {}
    for mood, mood_data in processed_data.items():
        final_data[mood] = {
            "pure": mood_data["pure"],
            "transitions": dict(mood_data["transitions"])
        }
    
    # Save processed data
    with open('data/processed_music_database.json', 'w') as file:
        json.dump(final_data, file, indent=2)

if __name__ == "__main__":
    process_music_database()