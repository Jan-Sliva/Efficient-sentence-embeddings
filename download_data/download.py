from urllib.request import urlretrieve
import gzip
import os

target_size = 500_000

with open("langs.txt", "r") as f:
    langs = f.readlines()

langs = list(map(lambda x: x.rstrip(), langs))

path = ""



for l in langs:

    line_count = 0
    year = 2023
    while(True):
        ziped_file = path + l + "_" + str(year) + ".gz"
        res_file = path + l + ".txt"

        url = "https://data.statmt.org/news-crawl/{}/news.{}.{}.shuffled.deduped.gz".format(l, year, l)
        try:
            urlretrieve(url, ziped_file)
            print("Download succefull from " + url)
        except Exception:
            print("Can't download from " + url)
            print("Continuing to next language (or ending)")
            break

        with gzip.open(ziped_file, 'rb') as f_in:
            with open(res_file, 'ab') as f_out:
                for line in f_in:
                    line = line.rstrip()
                    if line != "":
                        f_out.write(line)
                        f_out.write(bytes("\n", encoding="utf-8"))
                        line_count += 1
                    if line_count >= target_size:
                        break
                    
        os.remove(ziped_file)

        if line_count >= target_size:
            break

        year -= 1
