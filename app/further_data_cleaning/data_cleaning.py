import os
import re

dataorig_prefix = "./data_orig/"
dataorigbysect_prefix = "./data_orig_by_sect/"
datasave_prefix = "./data_by_sect/"


doc_names = os.listdir(datasave_prefix)

print("\n\n\n--------------------------------------------------------------\n")
print("\n--------------------------------------------------------------\n\n\n")

total = 0
for directory in doc_names:
    print(directory)
    dir_path = datasave_prefix+directory
    def_check = []
    stit_check = []
    comm_check = []
    app_check = []
    for sect_filename in os.listdir(dir_path):
        save_txt = None
        orig_text = None
        with open(dir_path+"/"+sect_filename) as sect_file:
            orig_text = sect_file.read().strip()
            inner_txt = orig_text.lower()
            if inner_txt.startswith(("definitions.", "dictionary.", "interpretation.")):
                def_check.append(sect_filename)
            if inner_txt.startswith("short title."):
                stit_check.append(sect_filename)
            if inner_txt.startswith(("commencement.", "commencement of act.")):
                comm_check.append(sect_filename)
            if inner_txt.startswith(("application of the ", "application to the ", "application of this act.", "application to this act.")):
                app_check.append(sect_filename)
            save_txt = re.sub(r'[-]{2,}', " ", orig_text)
            save_txt = re.sub(r'[\.]+', ".", save_txt)
            save_txt = re.sub(r'[\s]+', " ", save_txt)
            save_txt = re.sub(r'\)\(', ") (", save_txt)
        if save_txt:
            with open(dir_path+"/"+sect_filename, 'w') as sect_file:
                sect_file.write(save_txt)
            
    if def_check:
        print("Has definitions at", def_check)
    if stit_check:
        print("Has short title at", stit_check)
    if comm_check:
        print("Has commencement at", comm_check)
    if app_check:
        print("Has application at", app_check)
    print()
    total += len(def_check) + len(stit_check) + len(comm_check) + len(app_check)

    for to_del in set(def_check + stit_check + comm_check + app_check):
        os.remove(dir_path+"/"+to_del)

print(total)