"""
This module provides a script to download and preprocess data from the News Crawl dataset.

usage:
python download.py --path <path_to_data_folder> --target_size <number_of_lines_to_save_per_language>
"""
from urllib.request import urlretrieve
import gzip
import os
import os.path as P
import argparse

def main():
    parser = argparse.ArgumentParser(description='Download and preprocess data from the News Crawl dataset.')
    parser.add_argument('--path', type=str, required=True, help='Path to save data')
    parser.add_argument('--target_size', type=int, default=500_000, help='Number of lines to save per language')
    args = parser.parse_args()

    if not P.exists(args.path):
        os.mkdir(args.path)

    with open(P.join("download_data", "langs.txt"), "r") as f:
        langs = f.read().splitlines()

    for l in langs:
        line_count = 0
        year = 2023
        while True:
            ziped_file = P.join(args.path, f"{l}_{year}.gz")
            res_file = P.join(args.path, f"{l}.txt")

            url = f"https://data.statmt.org/news-crawl/{l}/news.{year}.{l}.shuffled.deduped.gz"
            try:
                urlretrieve(url, ziped_file)
                print(f"Download successful from {url}")
            except Exception:
                print(f"Can't download from {url}")
                print("Continuing to next language (or ending)")
                break

            with gzip.open(ziped_file, 'rb') as f_in:
                with open(res_file, 'ab') as f_out:
                    for line in f_in:
                        line = line.rstrip()
                        if line:
                            f_out.write(line)
                            f_out.write(b"\n")
                            line_count += 1
                        if line_count >= args.target_size:
                            break
                    
            os.remove(ziped_file)

            if line_count >= args.target_size:
                break

            year -= 1

if __name__ == "__main__":
    main()
