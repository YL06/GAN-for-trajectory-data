from selenium import webdriver
import time

output_dir = 'output'
url = 'file:D:/BA/Two-Stage-Gan-in-trajectory-generation/pre_process/map_generation/map.html'
brower = webdriver.Chrome('C:/Users/YJ/Downloads/chromedriver_win32/chromedriver.exe')
brower.get(url)
brower.maximize_window()
start_time = time.time()
time.sleep(0.2)
for i in range(130):
    name = str(i)
    if len(name)<3:
        name = '0'*(3-len(name))+name
    brower.save_screenshot('output/pict28_%s.png'%(name))
    print("--- %s seconds ---" % (time.time() - start_time))
    time.sleep(2.9)
brower.close()
