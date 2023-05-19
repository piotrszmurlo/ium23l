import pandas as pd
import matplotlib.pyplot as plt

workdir = 'data'
artists = pd.read_json(workdir + '/artists.jsonl', lines=True)
tracks = pd.read_json(workdir + '/tracks.jsonl', lines=True)

artists_tracks = artists.merge(tracks, left_on='id', right_on='id_artist', how='left')
tracks_count = artists_tracks.groupby(['id_x'])['id_y'].count().sort_values()
tracks_count.value_counts().plot.bar(width=1.0)
plt.show()
leaders = tracks_count[tracks_count > 100]
print(leaders)
print(f'Artists with over 100 songs {len(leaders)}, that makes {100 * len(leaders) / len(tracks_count)}%')


artists_genres = artists.explode('genres').reset_index(drop=True)
artists_genres.genres.value_counts().plot.bar(width=1.0)
plt.tick_params(labelbottom=False)
plt.show()

leaders_genres = artists_genres.merge(leaders, left_on='id', right_on='id_x', how='right')
leaders_genres_count = leaders_genres.groupby(['id'])['genres'].count()
print(leaders_genres_count)
