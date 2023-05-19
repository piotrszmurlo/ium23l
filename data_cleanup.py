import pandas as pd
from IPython.core.display_functions import display

from artists_cleanup import get_cleaned_up_artists_data


def main():
    artists_data = pd.read_json("data/artists.jsonl", lines=True)
    tracks_data = pd.read_json("data/tracks.jsonl", lines=True)
    min_occurrences = 100
    display(get_clean_merged_tracks_data(artists_data, tracks_data, min_occurrences))


# główna funkcja zwracająca wyczyszczone dane
def get_clean_merged_tracks_data(artists_data, tracks_data, min_occurrences):
    labels_to_drop = ['name', 'id', 'release_date', 'mode']
    tracks_data.drop(labels_to_drop, axis=1, inplace=True)
    clean_artists_data = get_cleaned_up_artists_data(artists_data, min_occurrences)
    merged_tracks_data = tracks_data.merge(clean_artists_data, left_on='id_artist', right_on='id', how='inner')
    labels_to_drop = ['id_artist', 'id']
    merged_tracks_data.drop(labels_to_drop, axis=1, inplace=True)
    merged_tracks_data.rename(columns={'genres': 'genre'}, inplace=True)
    return merged_tracks_data


if __name__ == '__main__':
    main()
