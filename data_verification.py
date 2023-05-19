import pandas as pd

workdir = 'data'
artists = pd.read_json(workdir + '/artists.jsonl', lines=True)
tracks = pd.read_json(workdir + '/tracks.jsonl', lines=True)

wrong_artist_id = artists[artists.id == -1]
print(f'{len(wrong_artist_id)} artists have an id of -1, that makes {100* len(wrong_artist_id) / len(artists)}% of all')

wrong_artist_name = artists[artists['name'].isnull()]
print(f'{len(wrong_artist_name)} artists have no name, that makes {100* len(wrong_artist_name) / len(tracks)}% of all')

wrong_track_id = tracks[tracks['id'].isnull()]
print(f'{len(wrong_track_id)} tracks have an id of null, that makes {100* len(wrong_track_id) / len(tracks)}% of all')

wrong_track_artist_id = tracks[tracks['id_artist'].isnull()]
print(f'{len(wrong_track_artist_id)} tracks are not connected to an artist, that makes {100* len(wrong_track_artist_id) / len(tracks)}% of all')

wrong_track_name = tracks[tracks['name'].isnull()]
print(f'{len(wrong_track_name)} tracks have an id of null, that makes {100* len(wrong_track_name) / len(tracks)}% of all')


artists_tracks = artists.merge(tracks, left_on='id', right_on='id_artist', how='left')
artists_without_tracks = len(artists_tracks[artists_tracks['energy'].isna()])
print(f'{artists_without_tracks} artists have no songs, that makes {100* artists_without_tracks / len(artists)}% of all')
