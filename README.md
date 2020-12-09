# aai4r-Situation-Recognition

### Install requirments & download files
To install requirments, refer package-list.txt file.

Download files as below structure.

    .
    ├── Dataset
    │   └── imsitu 
    │       └── of500_images_resized
    ├── verb_vocabulary_v1.pickle
    ├── role_vocabulary_v1.pickle
    ├── noun_vocabulary_v1.pickle
    └── output_crf_v1
        ├── best.model
        └── encoder
### Images download link
https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar

### pickle files and output_crf_v1 folder download link
https://drive.google.com/drive/u/0/folders/1I75NzFC18XQjYrbz9XQXZiWMAoSyDax1?ths=true

### Evaluate
python ggnn.py

### Train
Revise train epoch in ggnn.py file.

Note: This will overwrite current model_best.pth.tar file.
