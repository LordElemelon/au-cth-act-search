from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Firefox()

# all data downloaded from www.austlii.edu.au

driver.get("http://www8.austlii.edu.au/cgi-bin/viewdb/au/legis/cth/consol_act/")
word_list_elem = driver.find_element_by_xpath("//*[@id='panel-letter']/ul")
letter_links = [e.get_attribute("href") for e in word_list_elem.find_elements_by_tag_name("a") if "Any" not in e.text]
filtered_list = []
gottem = False
for hh in letter_links:
    if "toc-P" in hh:
        gottem = True
    if gottem:
        filtered_list.append(hh)

for letter_link in filtered_list:

    driver.get(letter_link)
    
    doc_list = driver.find_element_by_xpath("//*[@id='page-main']/div/div/ul").find_elements_by_tag_name("a")
    link_names = [e.text for e in doc_list]
    links = [e.get_attribute("href") for e in doc_list]

    # for j, k in zip(link_names, links):
    #     print(j)
    #     print(k)
    #     print("\n")

    for link, link_name in zip(links, link_names):
        filename = "data/" + link_name.replace(" ", "_").replace("\"", "") + ".txt"
        driver.get(link)

        element = driver.find_element_by_class_name("side-download")
        dl_links = [e for e in element.find_elements_by_tag_name("a") if "Plain" in e.text]
        if not dl_links:
            print("somethings wrong I can feel it")
            print(link)
            print([e.text for e in element.find_elements_by_tag_name("a")])
            continue
        dl_link = dl_links[0].get_attribute("href")

        driver.get(dl_link)

        a_tags = WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )
        dl_links = [e for e in driver.find_elements_by_tag_name("a") if "Plain" in e.text]
        if not dl_links:
            print("somethings wrong I can feel it again heh")
            print(link)
            print([e.text for e in driver.find_elements_by_tag_name("a")])
            continue
        dl_link = dl_links[0].get_attribute("href")

        driver.get(dl_link)
        a_tags = WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.TAG_NAME, "pre"))
        )
        full_txt = driver.find_element_by_tag_name("pre").text
        with open(filename, "w") as text_file:
            print(full_txt, file=text_file)

