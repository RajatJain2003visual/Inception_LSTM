import functions
from functions import initialize,generate_captions_beam_search,vocabulary
import torch
from PIL import Image
import streamlit as st

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab,model,exist = initialize(device)

st.title("Image captioning")
labeld = st.header("Upload file")

def initialize_selected_image():
  # print("runned")
  st.session_state['selected_image'] = None

def caption_image(image_path,beam_size):
  captions = generate_captions_beam_search(model, image_path, beam_size=beam_size, device=device)
  text = []
  for i,caption in enumerate(captions):
    caption1 = " ".join(caption[1])
    text.append(caption1)
  return text

if "selected_image" not in st.session_state:
  initialize_selected_image()

# Sidebar for sample images
with st.sidebar:
  st.title("Sample Images")
  with st.container():
    st.image("sample_images/1.jpg", width=100)
    if st.button("Generate Caption",key="1"):
      st.session_state['selected_image'] = "1.jpg"

  with st.container():
    st.image("sample_images/2.jpg", width=100)
    if st.button("Generate Caption",key="2"):
      st.session_state['selected_image'] = "2.jpg"

  with st.container():
    st.image("sample_images/3.jpg", width=100)
    if st.button("Generate Caption",key="3"):
      st.session_state['selected_image'] = "3.jpg"

  with st.container():
    st.image("sample_images/4.jpg", width=100)
    if st.button("Generate Caption",key="4"):
      st.session_state['selected_image'] = "4.jpg"

uploaded_file = st.file_uploader(label="Choose a file", type=["jpeg","jpg","png","webp"])
if uploaded_file:
  st.session_state['selected_image'] = None
  st.success("File uploaded")

if uploaded_file is not None:
  save_path = "test_img.jpg"
  with open(save_path, "wb") as f:
      f.write(uploaded_file.getbuffer())
  st.success(f"File saved to: {save_path}")

if st.session_state['selected_image'] is not None or uploaded_file is not None:
  if st.session_state['selected_image'] is not None:
    path = "sample_images/"+st.session_state['selected_image']
  else:
    path = "test_img.jpg"

  st.image(path, width=700)
  beam_size = st.slider("Select the beam size", 1, 10, value=3)
  captions = caption_image(path,beam_size)
  for caption in captions:
    with st.container(border=True):
      st.write(caption)

