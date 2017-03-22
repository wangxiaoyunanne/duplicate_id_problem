from urllib2 import urlopen
#from robobrowser import RoboBrowser
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import numpy as np
############# input is str not int
def primary_id(fb_id):
	driver = webdriver.Firefox()
	driver.get("https://www.facebook.com")
	elem = driver.find_element_by_name("email")
	elem.send_keys("annewilson816@gmail.com")#facebook username
	#elem.clear()
	elem = driver.find_element_by_name("pass")#facebook password
	#elem2.clear()
	elem.send_keys("wangxy37")
	#driver.implicitly_wait(5)
	elem.send_keys(Keys.RETURN)
	#elem = driver.find_element_by_css_selector(".selected")
	#print ("test")
	elem.click()
	time.sleep(5) # make a deley
	#url1 = '100002065534410'
	url = "https://www.facebook.com/" + fb_id
        print url
        driver.get(url)
	r_id=driver.current_url#.replace("https://www.facebook.com/", "")
	#print (url)
	#print (r_id)
	driver.quit()
	return r_id[25: ]
####################################################
# when the input is a weighed projected graph
result = pd.read_table("Downloads/new.txt",sep = ',')
result['truth'] = pd.Series (np.zeros(len(result)))
#i = '964255970322560'
result['ID'] = result['ID'].astype('str')
for i in range (len(result)): 
    result['truth'][i] = primary_id(result['ID'][i])
#print isblocked(i)
result.to_csv('Desktop/sincere data/id_cons.txt', index = False)
#####################################################
# when the input is a bipartite graph
result = pd.read_table("Downloads/name_id_49070.txt",sep = '\t', header = None)
# get the unique users IDs
ID_list =  np.unique(result.ix[:,0])
ID_list = ID_list.astype('str')
output = []
for i in ID_list:
    truth = primary_id(i)
    output.append([i,truth])

output =  pd.DataFrame(output)
result.to_csv('Desktop/sincere data/id_cons_49070.txt', index = False)

