import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import subprocess
import os
import shutil

st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")
st.header("Filters")

def main():
    #uploaded_file = st.file_uploader("Choose a file")
    file_uploaded = st.file_uploader("Choose the file", type=['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        if os.path.isdir('C:/Users/HP/Desktop/Minor/images/') == True:
          #  print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            shutil.rmtree('images')
            os.mkdir('images')

        image = Image.open(file_uploaded)
        image.save(f"images/s11.jpeg")
        st.image(image, 'Content' ,width = 300)
        figure = plt.figure()
        plt.imshow(image)
    file1_uploaded = st.file_uploader("Choose Filter", type=['jpg', 'png', 'jpeg'])
    if file1_uploaded is not None:
      image = Image.open(file1_uploaded)
      image.save(f"images/s12.png")
      st.image(image, 'Style',width=300)
      figure = plt.figure()
      plt.imshow(image)
      cmd = 'python NST.py'
      p=subprocess.Popen(cmd,shell=True)
      out,err=p.communicate()
      print(err)
      print(out)
      img = Image.open('images/image_2500.jpeg')
      st.image(img, 'Final image',width=300)
     # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


main()




