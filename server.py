import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import os
from scipy.spatial import cKDTree
import pickle
from classification import yoloclas

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()

ind2word = {}
with open('id2class.txt', 'r') as f:
    for line in f:
        idx, class_name = line.strip().split(' ')
        ind2word[class_name] = idx

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        cls=yoloclas(uploaded_img_path)
        # Run search
        query = fe.extract(img)
        print(ind2word[cls])
        kd_path = os.path.join("./static/imagenetkdt",ind2word[cls]+'_kd_tree.pkl')
        print(kd_path)
        with open(kd_path, 'rb') as f:
            kd_tree, img_paths = pickle.load(f)
        distances, indices = kd_tree.query(query, k=20)  # k=10 for the 10 closest neighbors
        scores = [(distances[i], os.path.join("/static/imagenet-mini/train",img_paths[index])) for i, index in enumerate(indices)]
        print(scores[1])
        # dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        # ids = np.argsort(dists)[:20]  # Top 30 results
        # scores = [(dists[id], img_paths[id]) for id in ids]
        if scores[0][0]>70:
            scores.clear()
            cls=None

        return render_template('search.html',
                               query_path=uploaded_img_path,
                               scores=scores,
                               cls=cls)
    else:
        return render_template('search.html')


if __name__=="__main__":
    app.run("0.0.0.0")
