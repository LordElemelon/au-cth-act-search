import os
import re

dataorig_prefix = "./data_orig/"
dataorigbysect_prefix = "./data_orig_by_sect/"
datasave_prefix = "./data_by_sect/"

# \s+\n[0-9][0-9a-zA-Z]*  ------ find section
# \s+\n[a-zA-Z].+?(a|A)mendment------ end has amendments
# \s+\n(t|T)he (s|S)chedule.*\s?\n------ end has schedule
# \s+\n(s|S)chedules?\s*.*\s?\n------ end schedule
# \s+\n(n|N)otes to ------ end notes
# \s+\n[a-zA-Z].*( )*\s?\n.*\s?\n------ part/division find


orig_filenames = os.listdir(dataorig_prefix)
#print(orig_filenames)

#orig_filenames = [orig_filenames[0]]
pic_ones = []
nonpic_ones = []

for filename in orig_filenames:
    # print(filename)
    with open(dataorig_prefix+filename) as orig_file:
        orig_text = orig_file.read()
        if orig_text.startswith("[pic]"):
            pic_ones.append(filename)
            # print("\n\n__FILE__  ",filename,"\n")
            # for jo in orig_text.split("\n")[0:10]:
            #     print(jo)
        else:
            nonpic_ones.append(filename)
            # print(orig_text.find("TABLE OF PROVISIONS"))
            # if orig_text.find("TABLE OF PROVISIONS") == 0 or orig_text.find("TABLE OF PROVISIONS") > 250:
            #     print(filename)
            # print("\n\n__FILE__  ",filename,"\n")
            # for jo in orig_text.split("\n")[0:10]:
            #     print(jo)
print()
print(len(pic_ones))
print(len(nonpic_ones))

for filename in nonpic_ones:
    print(filename[:-4])
    if not os.path.exists(datasave_prefix+filename[:-4]):
        os.makedirs(datasave_prefix+filename[:-4])
    if not os.path.exists(dataorigbysect_prefix+filename[:-4]):
        os.makedirs(dataorigbysect_prefix+filename[:-4])
    with open(dataorig_prefix+filename) as orig_file:
        orig_text = orig_file.read()
        arr_text = orig_text.split("- SECT ")
        # print(len(arr_text))
        file_title = arr_text[0][0:arr_text[0].find("\n")].strip()

        section_titles = []
        section_texts = []
        # i = 0
        for section in arr_text[1:]:
            split_point = section.find("\n")
            section_title = section[:split_point]
            section_text = section[split_point+1:section.find(file_title)].strip()

            with open(dataorigbysect_prefix+filename[:-4]+"/"+section_title+".txt", "w") as text_file:
                print(section_text, file=text_file)

            dot_addition_pointers = [(x.start(), x.end()) for x in re.finditer(r'\s+\n[^\"\'\s]+[^\n\.\:\;]+\n\s*\n', section_text)]
            # print(dot_addition_pointers)
            for s in dot_addition_pointers:
                # print(">>" + section_text[s[1]-20:s[1]] + "<<")
                # print(">>" + section_text[s[1]-1] + "<<")
                to_add = s[1]-re.search(r'\S', section_text[s[0]:s[1]][::-1]).end()+1
                # print(section_text[to_add-6:to_add]+"."+section_text[to_add+1:to_add+20])
                section_text = section_text[:to_add] + "." + section_text[to_add+1:]

            first_nl = section_text.find("\n")
            section_text = section_text[:first_nl] + "." + section_text[first_nl+1:]

            # print(section_title, "\n\n-----\n")
            # print(section_text, "\n\n-----\n")
            
            new_text = re.sub(r'\n(\s)*\((\w){1,5}\)\s', ' ', section_text)
            new_text = re.sub(r'\s+', ' ', new_text)

            with open(datasave_prefix+filename[:-4]+"/"+section_title+".txt", "w") as text_file:
                print(new_text, file=text_file)

            # print(new_text)
            # print("\n----------------------------\n")
            # i += 1
            # if i > 6:
            #     break
        # print(arr_text[0], "\n\n\n")
        # print(arr_text[-1])
    
for filename in pic_ones:
    print(filename[:-4])
    if not os.path.exists(datasave_prefix+filename[:-4]):
        os.makedirs(datasave_prefix+filename[:-4])
    if not os.path.exists(dataorigbysect_prefix+filename[:-4]):
        os.makedirs(dataorigbysect_prefix+filename[:-4])
    with open(dataorig_prefix+filename) as orig_file:
        orig_text = orig_file.read()
        igind = re.search(r'\nContents\s\s*(.|\n)+?\s+\n[0-9][0-9a-zA-Z]*  ', orig_text)

        if not igind:
            print("- - - - - NADA - - - - - -\n\n")
            continue

        temp_calc = igind.end() - orig_text[igind.start():igind.end()][::-1].find("\n")
        cut_text = "\n\n" + orig_text[temp_calc:]
        # print(cut_text[:35])

        prob_end_1 = re.search(r'\s*\n(t|T)he (s|S)chedule.*?\s?\n', cut_text[:])
        prob_end_2 = re.search(r'\s*\n(s|S)chedules?\s*.*?\s?\n', cut_text[:])
        prob_end_3 = re.search(r'\s*\n(n|N)otes to ', cut_text[:])
        
        prob_end_points = []

        if prob_end_1:
            prob_end_points.append(prob_end_1.start())
        if prob_end_2:
            prob_end_points.append(prob_end_2.start())
        if prob_end_3:
            prob_end_points.append(prob_end_3.start())
        
        # print(prob_end_points)

        if not prob_end_points:
            print("- - - - - NADA AGAIN - - - - - -\n\n")
            continue

        end_point = min(prob_end_points)
        cut_text = cut_text[:end_point] + "\n"

        no_table_text = re.sub(r'\n\|.+\|', "\n", cut_text)
        no_paren_text = re.sub(r'\[.+\]', "", no_table_text)

        no_part_text = no_paren_text

        for _ in range(3):
            no_part_text = re.sub(r'\s+\n[a-zA-Z].*( )*\n.*\s?\n', "\n\n\n", no_part_text)
            
        split_text = [x.strip() for x in re.split(r'(\s+\n[0-9][0-9a-zA-Z]*  )', no_part_text) if len(x.strip())]

        section_titles = []
        section_texts_pre = []

        for i,s in enumerate(split_text):
            section_texts_pre.append(s+"\n\n") if i%2 else section_titles.append(s)

        # for i in range(min(4, len(section_titles))):
        #     print(">"+section_titles[i]+"<")
        #     print(">"+section_texts_pre[i]+"<\n")

        for section_title, section_text_pre in zip(section_titles, section_texts_pre):
            section_text = section_text_pre

            with open(dataorigbysect_prefix+filename[:-4]+"/"+section_title+".txt", "w") as text_file:
                print(section_text, file=text_file)

            for dot_point in [x.start()+1 for x in re.finditer(r'[a-zA-Z0-9]\s?\n\s*\n', section_text_pre)]:
                section_text = section_text[:dot_point] + "." + section_text[dot_point+1:]
            section_text = re.sub(r'\n(\s)*\((\w){1,5}\)\s+', ' ', section_text)
            section_text = re.sub(r'\n(\s)*\"\((\w){1,5}\)\s+', ' \"', section_text)
            section_text = re.sub(r'\s+', ' ', section_text)
            section_text = section_text.strip()

            with open(datasave_prefix+filename[:-4]+"/"+section_title+".txt", "w") as text_file:
                print(section_text, file=text_file)
            # print(">"+section_title+"<")
            # print(">"+section_text+"<\n")



