This project uses a 3D CNN to detect lung cancer from 3D CT cubes generated from the LIDC-IDRI dataset.

The model is trained to classify 3D CT cubes of size 48x48x48 as:

1 Positive (cancerous)
0 Negative (healthy)


1 Preprocess LIDC-IDRI CT scans into 3D cubes
2 Train a 3D CNN model on the extracted cubes
3 Evaluate the trained model (confusion matrix + metrics)
4 Generate result figures (confusion matrix, ROC curve, PR curve)

 REQUIREMENTS:

 Windows 10 (tested) or Windows 11
 Python 3.10+
 LIDC-IDRI dataset 

python packages:

 numpy
 scipy
 scikit-learn
 matplotlib
 pylidc
 tensorflow (Windows users may use tensorflow-directml)
