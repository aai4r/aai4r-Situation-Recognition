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

### Online demo

Use pretrained model "model_best_online.pth.tar" [google drive link](https://drive.google.com/file/d/1R8J1Gc70fyKYTM5NZ5x8N9J_DQppweDE/view?usp=sharing)

You can change model to "model_best.pth.tar" after train by ggnn.py
  * change this line
  <pre><code>gnn.load_state_dict(torch.load('model_best_online.pth.tar')['state_dict'])</code></pre>
  <pre><code>gnn.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])</code></pre>

1. python online_demo_imgcv.py
  * check output images in testimg>output
  
2. python online_demo_framecv.py
  * online demo by webcam
