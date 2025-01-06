"""
This script processes and cleans a music dataset, generates spectrograms,
and organizes the data into training and testing sets.

Usage:
    python script.py [options]

Options:
    -n, --fit_amount (int)         Number of tracks to sample from each genre. Default is 1000.
    -d, --dataset_location (str)   Path to the dataset location. Default is "..\\..\\data".
    -b, --file_name (str)          Name of the file to save the cleaned dataset. Default is "songs".
    -t, --test_split (float)       Fraction of the data to be used as the test set. Default is 0.2.
    --genre_fill (bool)            Whether to fill missing genres. Default is False.
    -gu, --genres_unambigous       List of unambiguous genres to drop. Default is None.
    -r, --random_split (int)       Random seed for reproducibility. Default is 42.

Example:
    python script.py -n 1500 -d "../../data" -b "cleaned_songs" -t 0.25 --genre_fill True -gu "rock" "pop" -r 123

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import shutil
import argparse
import conv_img as conv
import os.path as pth
import os

def parse_arguments():
    """
    Parse command-line arguments.

    This function parses the command-line arguments provided by the user.

    Arguments:
        -n, --fit_amount (int): Number of tracks to sample from each genre. Default is 1000.
        -d, --dataset_location (str): Path to the dataset location. Default is "..\\..\\data".
        -b, --file_name (str): Name of the file to save the cleaned dataset. Default is "songs".
        -t, --test_split (float): Fraction of the data to be used as the test set. Default is 0.2.
        --genre_fill (bool): Whether to fill missing genres. Default is False.
        -gu, --genres_unambigous (list of str): List of unambiguous genres to drop. Default is None.
        -r, --random_split (int): Random seed for reproducibility. Default is 42.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--fit_amount', type=int, default=None)
    parser.add_argument('-d', '--dataset_location', type=str, default="..\\..\\data")
    parser.add_argument('-b', '--file_name', type=str, default="songs")
    parser.add_argument('-t', '--test_split', type=float, default=0.2)
    parser.add_argument('-s', '--subset', type=str, required=False, default="large")
    parser.add_argument('-gu', '--genres_unambigous', nargs='*', type=str, required=False, default=[])
    parser.add_argument('-r', '--random_split', type=int, default=43)
    parser.add_argument('--use_mfcc', action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def construct_dataframe(dataset_location:str) -> pd.DataFrame:
    """
    Create a dataset from the given CSV file.

    This function reads the CSV file from the specified location,
    filters it, and constructs a DataFrame out of it.

    Args:
        dataset_location (str): Path to the data location.

    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    
    all_tracks = pd.read_csv(dataset_location + "/tracks.csv", low_memory=False)
    labels=all_tracks.values
    label_genre_top=np.where(labels[0,:]=="genre_top")
    label_genres=np.where(labels[0,:]=="genres")
    label_subset=np.where(labels[0,:]=="subset")
    label_bit_rate=np.where(labels[0,:]=="bit_rate")


    all_id=labels[2:, 0]
    all_label_subset=labels[2:,label_subset[0][0]]
    all_genre_top=labels[2:,label_genre_top[0][0]]
    all_genres=labels[2:,label_genres[0][0]]
    all_bit_rate=labels[2:,label_bit_rate[0][0]]

    max_id_len=6
    path_array=[]
    folder_path=dataset_location + "\\fma_large"
    for id in all_id:
        id_len=len(str(id))
        formatted_id=str(id)
        if id_len!=max_id_len:
            formatted_id=str(id).zfill(max_id_len)
        st=formatted_id[0:3:1]
        sub_folder_path="\\"+st+"\\"+formatted_id+".mp3"
        mp3_path=folder_path+sub_folder_path
        path_array.append(mp3_path)

    songs=pd.DataFrame({
        "id":all_id,
        "genre":all_genre_top,
        "genres":all_genres,
        "subset":all_label_subset,
        'bit_rate': all_bit_rate,
        'path':path_array
    })
    return songs

def clean_dataframe(df:pd.DataFrame, fit_amount:int, dataset_location:str, use_mfcc:bool, subset:str) -> pd.DataFrame:
    """
    Clean the DataFrame by filtering and sampling tracks.

    This function drops rows with missing genres, filters genres with
    at least `fit_amount` tracks, samples `fit_amount` tracks from each
    genre, and updates the file paths.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        fit_amount (int): Number of tracks to sample from each genre.
        dataset_location (str): Path to the data location.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """

    if subset == "small":
        df = df[df["subset"] == "small"]
    elif subset == "medium":
        df = df[df["subset"].isin(["small", "medium"])]

    df = df.dropna(subset=["genre"])
    if fit_amount is not None:
        fitting_genres = [key for key, count in df["genre"].value_counts().items() if count >= fit_amount]
        df = df[df["genre"].isin(fitting_genres)]
        df = df.groupby('genre').apply(lambda x: x.sample(fit_amount, random_state=42)).reset_index(drop=True)
    df['genre'] = df['genre'].replace('Old-Time / Historic', 'Old-Time')

    

    df['path'] = df['path'].where(
        df['subset'].ne('small'),
        df['path'].replace('fma_medium', 'fma_large', regex=True)
    )

    df['path'] = df['path'].where(
        df['subset'].ne('medium'),
        df['path'].replace('fma_medium', 'fma_large', regex=True)
    )

    df['spectrogramPath'] = dataset_location + '\\spectrograms\\' + df['genre'] + '\\' + df['id'].astype('str') + '.png'

    if use_mfcc:
        df['mfccPath'] = dataset_location + '\\mfcc\\' + df['genre'] + '\\' + df['id'].astype('str') + '.npy'


    return df

def dataframe_drop_unambigous(df:pd.DataFrame, genres_unambigous:list) -> pd.DataFrame:
    """
    Drop rows with genres specified as unambiguous from
    the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        genres_unambigous (list): List of unambiguous genres to drop.

    Returns:
        pd.DataFrame: DataFrame with unambiguous genres dropped.
    """

    df = df[~df["genre"].isin(genres_unambigous)]
    return df


def create_folder_structure(df:pd.DataFrame, dataset_location:str, use_mfcc:bool) -> None:
    """
    Create the folder structure for storing spectrogram images.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        dataset_location (str): Path to the data location.

    Returns:
        None
    """

    spec_folders = dataset_location + '\\spectrograms\\' + df['genre'].unique()

    for genre in spec_folders:
        if not os.path.isdir(genre):
            os.makedirs(genre)

    if use_mfcc:
        mfcc_folders = dataset_location + '\\mfcc\\' + df['genre'].unique()

        for genre in mfcc_folders:
            if not os.path.isdir(genre):
                os.makedirs(genre)

def create_spectrograms(df:pd.DataFrame, use_mfcc:bool) -> list[str]:
    """
    Generate spectrograms for each track in the DataFrame.

    This function generates spectrogram images for each track in the
    DataFrame and saves them to the specified paths. It also collects
    a list of corrupted files.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
        list: List of IDs of corrupted files.
    """

    i = 1
    list_corrupted = []

    for _, entry in df.iterrows():
        print(f'progress: {i}/{len(df)}')
        print(f'file id: {entry["id"]} // genre: {entry["genre"]}')
        if not pth.exists(entry["spectrogramPath"]) or os.path.getsize(entry["spectrogramPath"]) == 0 or (use_mfcc and not pth.exists(entry["mfccPath"])):
            try:
                if use_mfcc:
                    conv.generate_spectrogram(entry["path"], entry["spectrogramPath"], use_mfcc, entry["mfccPath"])
                else:
                    conv.generate_spectrogram(entry["path"], entry["spectrogramPath"])
            except Exception as e:
                print(f"Error: {e}")
                list_corrupted.append(entry["id"])
        i += 1


    print(f"Number of corrupted files: {len(list_corrupted)}")
    return list_corrupted

def clean_corrupted_data(df:pd.DataFrame, list_corrupted:list, dataset_location:str, file_name:str) -> None:
    """
    Clean the DataFrame by removing corrupted files.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        list_corrupted (list): List of IDs of corrupted files.
        dataset_location (str): Path to the data location.
        file_name (str): Name of the file to save the cleaned dataset.

    Returns:
        None
    """
    print(f"Corrupted files: {list_corrupted}")
    df = df[~df['id'].isin(list_corrupted)]
    df.to_csv(dataset_location + "\\" + file_name + ".csv", index=False)
    return df

def create_image_csv(df:pd.DataFrame, dataset_location:str, col_name, file_name) -> pd.DataFrame:
    """
    Create a CSV file for desired column with images.

    This function creates a CSV file containing the paths to the
    images for each row in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
        dataset_location (str): Path to the data location.

    Returns:
        pd.DataFrame: DataFrame with image paths.
    """

    df = df.drop(['subset', 'genres', 'path'], axis=1)
    df = df.drop(df[~df[col_name].apply(os.path.exists)].index)
    df.reset_index()
    df.to_csv(dataset_location + f'/{file_name}.csv')
    return df

def organize_data(df: pd.DataFrame, dataset_location: str, train_dir: str, test_dir: str, test_size: int, random_state: int, use_mfcc: bool) -> None:
    """
    Organize data into training and testing sets.

    This function splits the data into training and testing sets,
    creates directories for each genre, and copies the spectrogram
    images to the corresponding directories.

    Args:
        train_data (pd.DataFrame): DataFrame containing the training data.
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the testing data directory.
        test_split (float): Fraction of the data to be used as the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        None
    """

    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['genre'])
    print(os.path.join(dataset_location, train_dir))
    create_directories(dataset_location, train_dir, test_dir)
    save_split_data(train_data, test_data, dataset_location, train_dir, test_dir)
    process_split_data(train_data, dataset_location, train_dir)
    process_split_data(test_data, dataset_location, test_dir)
    print(f"Data successfully split and organized into '{train_dir}' and '{test_dir}'.")


def create_directories(dataset_location: str, train_dir: str, test_dir: str) -> None:
    """
    Create directories for training and testing data.

    Args:
        dataset_location (str): Path to the data location.
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the testing data directory.
        use_mfcc (bool): Whether to use MFCC data.

    Returns:
        None
    """
    if not os.path.isdir(os.path.join(dataset_location, train_dir)):
        os.makedirs(os.path.join(dataset_location, train_dir))

    if not os.path.isdir(os.path.join(dataset_location, test_dir)):
        os.makedirs(os.path.join(dataset_location, test_dir))


def save_split_data(train_data: pd.DataFrame, test_data: pd.DataFrame, dataset_location: str, train_dir: str, test_dir: str) -> None:
    """
    Save the split data into CSV files.

    Args:
        train_data (pd.DataFrame): DataFrame containing the training data.
        test_data (pd.DataFrame): DataFrame containing the testing data.
        dataset_location (str): Path to the data location.
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the testing data directory.
        use_mfcc (bool): Whether to use MFCC data.

    Returns:
        None
    """
    train_csv = os.path.join(dataset_location, train_dir, "train_spectrograms.csv")
    test_csv = os.path.join(dataset_location, test_dir, "test_spectrograms.csv")
    train_data.to_csv(train_csv, index=False)
    test_data.to_csv(test_csv, index=False)


def process_split_data(split_data: pd.DataFrame, dataset_location: str, split_dir: str) -> None:
    """
    Process split data by creating directories and copying files.

    Args:
        split_data (pd.DataFrame): DataFrame containing the split data.
        dataset_location (str): Path to the data location.
        split_dir (str): Path to the split data directory.
        use_mfcc (bool): Whether to use MFCC data.

    Returns:
        None
    """
    for genre in split_data['genre'].unique():
        genre_path = os.path.join(dataset_location, split_dir, genre)
        os.makedirs(genre_path, exist_ok=True)

    for _, row in split_data.iterrows():
        copy_files(row, dataset_location, split_dir)
    


def copy_files(row: pd.Series, dataset_location: str, split_dir: str) -> None:
    """
    Copy spectrogram and MFCC files to the target directory.

    Args:
        row (pd.Series): Row of the DataFrame containing file paths.
        dataset_location (str): Path to the data location.
        split_dir (str): Path to the split data directory.
        use_mfcc (bool): Whether to use MFCC data.

    Returns:
        None
    """
    source_path = row['spectrogramPath']
    genre = row['genre']
    target_dir = os.path.join(dataset_location, split_dir, genre)
    target_path = os.path.join(target_dir, os.path.basename(source_path))

    if not os.path.isfile(source_path):
        print(f"Warning: File {source_path} not found. Skipping.")
        return
    shutil.copy(source_path, target_path)


def create_dataset(dataset_location, file_name, fit_amount, genres_unambigous, use_mfcc, subset) -> pd.DataFrame:
    """
    Create a cleaned and balanced dataset.

    This function constructs a DataFrame from the dataset location,
    cleans and balances the DataFrame by filtering and sampling tracks,
    drops unambiguous genres, and saves the cleaned DataFrame to a CSV file.

    Args:
        dataset_location (str): Path to the data location.
        file_name (str): Name of the file to save the cleaned dataset.
        fit_amount (int): Number of tracks to sample from each genre.
        genres_unambigous (list): List of unambiguous genres to drop.

    Returns:
        pd.DataFrame: Cleaned and balanced DataFrame.
    """
    df = construct_dataframe(dataset_location)
    df_balanced = clean_dataframe(df, fit_amount, dataset_location, use_mfcc, subset)
    df_ready = dataframe_drop_unambigous(df_balanced, genres_unambigous)
    df_ready.to_csv(dataset_location + "\\" + file_name + ".csv", index=False)
    return df_ready

def create_spectrogram_data(
        df:pd.DataFrame, dataset_location:str,
        file_name:str, test_split:float, random_state:int,
        use_mfcc:bool) -> None:
    """
    Create spectrogram data from the given DataFrame.

    This function creates the folder structure, generates spectrograms,
    cleans corrupted data, creates a CSV file for spectrograms, and
    organizes the data into training and testing sets.

    Args:
        df (pd.DataFrame): DataFrame containing the info of dataset.
        dataset_location (str): Path to the data location.
        file_name (str): Name of the file to save the cleaned dataset.
        test_split (float): Fraction of the data to be used as the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        None
    """
    
    create_folder_structure(df, dataset_location, use_mfcc)

    list_corrupted = create_spectrograms(df, use_mfcc)
    df = clean_corrupted_data(df, list_corrupted, dataset_location, file_name)
    
    df_spectrograms = create_image_csv(df, dataset_location, 'spectrogramPath', 'spectrograms')

    organize_data(
        df_spectrograms, dataset_location, 'spectrogramTRAIN',
        'spectrogramTEST', test_split, random_state, use_mfcc
    )


if __name__ == "__main__":
    args = parse_arguments()

    df_ready = create_dataset(
        args.dataset_location, args.file_name,
        args.fit_amount, args.genres_unambigous,
        args.use_mfcc, args.subset
    )
    print("dataset created...")

    create_spectrogram_data(
        df_ready, args.dataset_location, 
        args.file_name, args.test_split, 
        args.random_split, args.use_mfcc
    )

    print("spectrograms created and data organized...")
    print("done!")

