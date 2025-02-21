# Estimation of continuous valence and arousal levels from faces in naturalistic conditions, Nature Machine Intelligence 2021

========================================================

**Go to the bottom for instructions on installing the environment and running the code with a directory**

========================================================

Official implementation of the paper _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021 [[1]](#Citation).
Work done in collaboration between Samsung AI Center Cambridge and Imperial College London.

Please find a full-text, view only, version of the paper [here](https://rdcu.be/cdnWi).

The full article is available on the [Nature Machine Intelligence website](https://www.nature.com/articles/s42256-020-00280-0).

[Demo] Discrete Emotion + Continuous Valence and Arousal levels      |  [Demo] Displaying Facial Landmarks
:-------------------------------------------------------------------:|:--------------------------------------:
<img src='images/emotion_only.gif' title='Emotion' style='max-width:600px'></img>  |  <img src='images/emotion_with_landmarks.gif' title='Emotion with landmarks' style='max-width:600px'></img>


## Youtube Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=J8Skph65ghM">Automatic emotion analysis from faces in-the-wild
  <br>
  <img src="https://img.youtube.com/vi/J8Skph65ghM/0.jpg"></a>
</p>

## Updates

**August, 24th, 2024**: We added ```demo_video.py``` as an example of how to run face detection and emotion recognition on a video. The script includes a visualization similar to what is shown in our Youtube video. 

**August, 14th, 2024**: We added ```demo.py``` as an example of how to run the model on a single face image.

## Testing the pretrained models on an image/video

The code requires the following Python packages : 

```
  Pytorch (tested on version 1.2.0)
  OpenCV (tested on version 4.1.0
  skimage (tested on version 0.15.0)
  face alignment (https://github.com/1adrianb/face-alignment)
```

We provide two pretrained models : one on 5 emotional classes and one on 8 classes. In addition to categorical emotions, both models also predict valence and arousal values as well as facial landmarks.

To run the model on a single image you can use the following command:

```
  python demo.py --nclass 8 --image_path images/example.png
```

We also provide a script to run emotion recognition on a video. The script includes a visualization similar to what is shown in our Youtube video. To run the model on a given video, you can use the following command:

```
  python demo_video.py --nclass 8 --video_path relative_path_to_your_video.mp4 --output_path output.mp4
```

## Quantitatively testing the pretrained models

To evaluate the pretrained models on the cleaned AffectNet test set, you need to first download the [AffectNet dataset](http://mohammadmahoor.com/affectnet/). Then simply run : 

```
  python test.py --nclass 8
```

where nclass defines which model you would like to test (5 or 8).

Please note that the provided pickle files contain the list of images (filenames) that we used for testing/validation but not the image files.

The program will output the following results :

#### Results on AffectNet cleaned test set for 5 classes


```
 Expression
  ACC=0.82

 Valence
  CCC=0.90, PCC=0.90, RMSE=0.24, SAGR=0.85
 Arousal
  CCC=0.80, PCC=0.80, RMSE=0.24, SAGR=0.79
```

#### Results on AffectNet cleaned test set for 8 classes

```
  Expression
    ACC=0.75

  Valence
    CCC=0.82, PCC=0.82, RMSE=0.29, SAGR=0.84
  Arousal
    CCC=0.75, PCC=0.75, RMSE=0.27, SAGR=0.80
```

#### Class number to expression name

The mapping from class number to expression is as follows.

```
For 8 emotions :

0 - Neutral
1 - Happy
2 - Sad
3 - Surprise
4 - Fear
5 - Disgust
6 - Anger
7 - Contempt
```

```
For 5 emotions :

0 - Neutral
1 - Happy
2 - Sad
3 - Surprise
4 - Fear
```

## Citation

If you use this code, please cite:

```
@article{toisoul2021estimation,
  author  = {Antoine Toisoul and Jean Kossaifi and Adrian Bulat and Georgios Tzimiropoulos and Maja Pantic},
  title   = {Estimation of continuous valence and arousal levels from faces in naturalistic conditions},
  journal = {Nature Machine Intelligence},
  year    = {2021},
  url     = {https://www.nature.com/articles/s42256-020-00280-0}
}
```

[1] _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021 

## License

Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0 International Licence (CC BY-NC-ND) license.

========================================================
========================================================

## Conda environment installation

Open Anaconda prompt, navigate to the repo directory and create the recreate the environment with the file provided:

```
cd /path/to/emonet_repo
conda env create -f environment.yml
```

## Running video_dir_processing

Activate the environment

```
conda activate emonet_env
```

## Install cuda-enabled pytorch

You might need to do this manually depending on your gpu and cuda version. The command that worked last time we tried (20/02/2025) on Damien's PC with CUDA 12.4 was:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

**Note** that you might be able to get away with a pytorch for cpu installation if you change the script to run face_alignment with cpu. Open the scipt, Ctr-F cuda, uncomment the line saying 

device = 'cpu'

## TLDR

Running example:

```
python video_dir_processing.py
```

Running with your own data (and optional different output dir):

```
python video_dir_processing.py --video_dir_path path/to/videodir --output_dir_path path/to_outputdir
```

**Details on script**

You can inspect the script by typing

```
python video_dir_processing.py --help
```

Output

```
usage: video_dir_processing.py [-h] [--nclasses {5,8}]
                               [--video_dir_path VIDEO_DIR_PATH]
                               [--output_dir_path OUTPUT_DIR_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --nclasses {5,8}      Number of emotional classes to test the model on.
                        Please use 5 or 8.
  --video_dir_path VIDEO_DIR_PATH
                        Path to directory containing .mp4 videos to process.
  --output_dir_path OUTPUT_DIR_PATH
                        Path to directory where output videos and analysis
                        .csvs will be saved.
```


The script uses the 8-class emonet model by default. Run the script providing the directory where the .mp4 videos to be processed are stored using the --video_dir_path flag.
The output videos and files will be saved in the same input directory by default. 
You can set an output dir of your choice using the --output_dir_path flag should you need it to be different to the input one.

If you don't provide any flags, the script will use the videos in the example_videos directory.
You could replace the videos in this directory with your own ones but I would recommend leaving this as is and providing the directory when calling the script.
