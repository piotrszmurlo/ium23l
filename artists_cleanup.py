import pandas as pd
from IPython.display import display


def main():
    artists_data = pd.read_json("data/artists.jsonl", lines=True)
    clean = get_cleaned_up_artists_data(artists_data, 100)
    display(clean['genres'].value_counts())


def get_genre_occurrences(artists_data):
    genre_occurrences = {}
    for genres in artists_data['genres']:
        for genre in genres:
            genre_occurrences[genre] = genre_occurrences.setdefault(genre, 0) + 1
    return genre_occurrences


def get_most_occurring_genres_only_labels(artists_data, genre_occurrences):
    new_labels = []
    for genres in artists_data['genres']:
        if len(genres) == 1:
            new_labels.append(genres[0])
        else:
            most_popular = max(genres, key=lambda x: genre_occurrences[x])
            new_labels.append(most_popular)
    return new_labels


def initial_clean(artists_data):
    artists_data.drop('name', axis=1, inplace=True)
    artists_data['genres'] = artists_data['genres'].apply(
        lambda genre_list: [genre.strip().lower() for genre in genre_list])
    artists_data['genres'] = artists_data['genres'].apply(
        lambda genre_list: [genre.replace("hip hop", "hip-hop") for genre in genre_list])
    artists_data['genres'] = artists_data['genres'].apply(
        lambda genre_list: list(set(genre.split(" ")[-1] for genre in genre_list)))
    return artists_data


def get_cleaned_up_artists_data(artists_data, min_occurrence):
    artists_data = initial_clean(artists_data)
    new_genre_labels = get_most_occurring_genres_only_labels(artists_data, get_genre_occurrences(artists_data))

    artists_data["genres"] = new_genre_labels
    artists_data = artists_data.groupby('genres').filter(lambda x: len(x) >= min_occurrence)

    return artists_data


if __name__ == '__main__':
    main()
