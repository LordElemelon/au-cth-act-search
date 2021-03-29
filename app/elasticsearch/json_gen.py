import os
from .config import Config

if __name__ == '__main__':

    # JSON generation
    id_for_json = 1
    last_dir = None
    with open("./data1.json", "a+") as json_file:
        for directory in os.listdir(Config.corpus_path):
            if id_for_json > 37000:
                last_dir = directory
                break
            print(directory)
            dir_path = Config.corpus_path+"/"+directory
            for sect_filename in os.listdir(dir_path):
                with open(dir_path+"/"+sect_filename) as sect_file:
                    sect_text = sect_file.read().strip()
                    jsonish_string = '{"index":{"_id":'+str(id_for_json)+'}}\n{"name":"' \
                        + directory.replace('"', '') + '=' + sect_filename[:-4] + '","description":"' + sect_text.replace('"', '') + '"}\n'
                    json_file.write(jsonish_string)
                    id_for_json += 1
    found_flag = False
    with open("./data2.json", "a+") as json_file:
        for directory in os.listdir(Config.corpus_path):
            if not found_flag:
                if directory == last_dir:
                    found_flag = True
                continue
            print(directory)
            dir_path = Config.corpus_path+"/"+directory
            for sect_filename in os.listdir(dir_path):
                with open(dir_path+"/"+sect_filename) as sect_file:
                    sect_text = sect_file.read().strip()
                    jsonish_string = '{"index":{"_id":'+str(id_for_json)+'}}\n{"name":"' \
                        + directory.replace('"', '') + '=' + sect_filename[:-4] + '","description":"' + sect_text.replace('"', '') + '"}\n'
                    json_file.write(jsonish_string)
                    id_for_json += 1
