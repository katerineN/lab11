import PIL.Image
import cv2 as cv
import pandas as pd
import base64
from sklearn.neighbors import NearestNeighbors
import numpy as np
import streamlit as st
import PIL
import torch
from torchvision.models import resnet50

def preprocessing2(img):
  return torch.FloatTensor(cv.resize(img, (224, 224))).permute(2, 0, 1).unsqueeze(0) / 255.

def neiron_emb(img_path, model):
  img = cv.imread(img_path, cv.IMREAD_COLOR)
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  inp = preprocessing2(img)
  with torch.no_grad():
      out = model(inp)
  emb = out[0].numpy()
  return emb

@st.cache(allow_output_mutation=True)
def loadBases():
    emb_db = './embedBaseDf11.csv'
    emb_base = pd.read_csv(emb_db, delimiter=',')
    emb_base['embed'] = emb_base['embed'].apply(
        lambda x: np.frombuffer(base64.b64decode(bytes(x[2:-1], encoding='ascii')), dtype=np.float32))
    return emb_base


def encodeImage(arr) -> np.ndarray:
    emds = []
    for i in arr:
        emb, _ = np.histogram(i, 2048)
        emds.append(emb)
    return emds

def workWithDb(img, classifier, db, db_neighbours):
    predhist = neiron_emb(img, classifier)
    distances, indices = db_neighbours.kneighbors(predhist.reshape(1, -1), return_distance=True)
    print(indices)
    return db.loc[indices[0], ['path']].values, distances[0]
    #return db.loc[indices[0]-2, ['path']].values, distances[0]

def main():
    loaded_model = resnet50(True).eval()
    database = loadBases()
    db_neighbours = NearestNeighbors(n_neighbors=5, metric='cosine')
    # print(type(database))
    emdbs = encodeImage(database['embed'].values)
    db_neighbours.fit(emdbs, y=list(range(len(database))))
    up_file = st.file_uploader('Choice file', type=['jpeg', 'jpg', 'webp', 'png', 'tiff'])
    if up_file is not None:
        try:
            img = PIL.Image.open(up_file)
            img = PIL.ImageOps.exif_transpose(img)
            img_paths, dists = workWithDb(np.array(img), loaded_model, database, db_neighbours)
            print(img_paths)
            for i in range(len(img_paths)):
                st.image(PIL.Image.open('C:'+img_paths[i][0]),
                         caption='Image {} with dist {}'.format(i + 1, f'{dists[i]:.3f}', width=580))
        except Exception as e:
            st.write('CRASHED:{}'.format(e))

# main()

if __name__ == '__main__':
    main()