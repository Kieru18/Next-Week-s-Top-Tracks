import pandas as pd


DIR_DATA = '../data/v3'

FEATURE_SET = ['track_id', 'week', 'lag_like_count', 'lag_skip_count', 'lag_playtime_ratio', 
               'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'popularity', 'release_date',
               'duration_ms', 'danceability', 'acousticness', 'instrumentalness', 
               'play_count']

TRACK_FEATURES = ['track_id', 'popularity', 'duration_ms', 'explicit', 'danceability', 
                  'energy', 'key', 'loudness', 'speechiness', 'acousticness', 
                  'instrumentalness', 'liveness', 'valence', 'tempo', 'release_date']


def load_data():
    sessions = pd.read_csv(f'{DIR_DATA}/sessions.csv')
    tracks = pd.read_json(path_or_buf=f'{DIR_DATA}/tracks.jsonl', lines=True)
    tracks = tracks.rename(columns={'id': 'track_id'})
    tracks['release_date'] = pd.to_datetime(tracks['release_date'], errors='coerce')
    tracks['release_date'] = tracks['release_date'].dt.year

    return sessions, tracks


def get_merged_sessions_df(sessions, tracks):

    sessions_df = sessions.copy()
    sessions_df['timestamp'] = pd.to_datetime(sessions_df['timestamp'])
    sessions_df['week'] = sessions_df['timestamp'].dt.strftime('%U')

    start_date = pd.to_datetime('2020-12-28')

    sessions_df['week'] = (sessions_df['timestamp'] - start_date).dt.days // 7
    sessions_df['week'] = sessions_df['week'] + 1

    # getting rid of the first and last week, as their data is not complete
    sessions_df = sessions_df[sessions_df.week.isin([1, 158]) == False]

    merged_df = pd.merge(tracks[['track_id', 'duration_ms']], sessions_df, on='track_id', how='inner')

    return (merged_df, sessions_df)


def get_tw_counts(sessions_df, tracks):
    all_weeks = sorted(sessions_df['week'].unique())
    all_tracks = tracks['track_id'].unique()

    df1 = pd.DataFrame({'track_id': all_tracks[:]})
    df2 = pd.DataFrame({'week': all_weeks[:]})

    empty_track_week_df = pd.merge(df1, df2, how='cross')


    counts_df = sessions_df.groupby(['week', 'track_id'])['event_type'].value_counts().unstack(fill_value=0)
    counts_df = counts_df.rename(columns={'play': 'play_count', 'like': 'like_count', 'skip': 'skip_count'}).reset_index()

    tw_counts_df = pd.merge(empty_track_week_df, counts_df, on=['week', 'track_id'], how='left')
    tw_counts_df = tw_counts_df.fillna(0)

    return tw_counts_df


def calculate_meanplaytime(merged_df):
    playtime_df = merged_df[merged_df["event_type"] != "like"].copy()

    grouped_df = playtime_df.groupby(['week', 'track_id', 'user_id'])
    playtime_df['next_timestamp'] = grouped_df['timestamp'].shift(-1)
    playtime_df['next_event_type'] = grouped_df['event_type'].shift(-1)

    play_mask = (playtime_df['event_type'] == 'play') & (playtime_df['next_event_type'] == 'skip')
    play_duration = (playtime_df['next_timestamp'] - playtime_df['timestamp']).dt.total_seconds() * 1000
    track_duration = playtime_df['duration_ms']

    playtime_df['playtime_ratio'] = 1
    playtime_df.loc[play_mask, 'playtime_ratio'] = play_duration[play_mask] / track_duration[play_mask]

    playtime_df = playtime_df.drop(['next_event_type', 'next_timestamp'], axis=1)
    playtime_df = playtime_df[playtime_df["event_type"] != "skip"].reset_index()

    grouped_playtime = playtime_df[playtime_df['event_type'] == 'play'].groupby(['track_id', 'week'])
    mean_playtime = grouped_playtime['playtime_ratio'].mean().reset_index()

    return mean_playtime


def assemble_full_data(tw_counts_df, mean_playtime, tracks):
    all_counts_df = pd.merge(tw_counts_df, mean_playtime, on=['week', 'track_id'], how='left')
    all_counts_df = all_counts_df.fillna(0)

    # Add data for week that will be predicted in the future
    new_week = all_counts_df['week'].max() + 1
    all_tracks = tracks['track_id'].unique()
    new_week_df = pd.DataFrame({'track_id': all_tracks[:], 'week': new_week, 'like_count': 0, 'play_count': 0, 'skip_count': 0, 'playtime_ratio': 0})
    all_counts_df = pd.concat([all_counts_df, new_week_df], ignore_index=True)


    all_counts_df['lag_skip_count'] = all_counts_df.groupby('track_id')['skip_count'].shift(1)
    all_counts_df['lag_like_count'] = all_counts_df.groupby('track_id')['like_count'].shift(1)
    all_counts_df['lag_playtime_ratio'] = all_counts_df.groupby('track_id')['playtime_ratio'].shift(1)
    all_counts_df['lag_1'] = all_counts_df.groupby('track_id')['play_count'].shift(1)
    all_counts_df['lag_2'] = all_counts_df.groupby('track_id')['play_count'].shift(2)
    all_counts_df['lag_3'] = all_counts_df.groupby('track_id')['play_count'].shift(3)
    all_counts_df['lag_4'] = all_counts_df.groupby('track_id')['play_count'].shift(4)
    all_counts_df['lag_5'] = all_counts_df.groupby('track_id')['play_count'].shift(5)

    full_df = pd.merge(all_counts_df, tracks[TRACK_FEATURES], on=['track_id'], how='left')
    full_df = full_df.fillna(0)

    return full_df[FEATURE_SET]


def preprocess_data(sessions, tracks):
    merged_df, sessions_df = get_merged_sessions_df(sessions, tracks)
    tw_counts = get_tw_counts(sessions_df, tracks)
    mean_playtime = calculate_meanplaytime(merged_df)
    print("generating datframe")
    dataset = assemble_full_data(tw_counts, mean_playtime, tracks)
    return dataset


def generate_csv(dataFrame):
    dataFrame.to_csv('../data/preprocessed_data.csv', index=False)


if __name__ == "__main__":
    print('started')
    sessions, tracks = load_data()
    print('loaded')
    dataset = preprocess_data(sessions, tracks)
    print('saving to file')
    generate_csv(dataset)
    print('file saved')
