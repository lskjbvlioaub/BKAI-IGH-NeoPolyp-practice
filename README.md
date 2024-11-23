# Installation
Clone the repository and install required libraries.
  ```sh
!git clone https://github.com/lskjbvlioaub/bkai-igh-neopolyp-practice.git
!pip install -r /kaggle/working/bkai-igh-neopolyp-practice/requirements.txt
  ```
Directing to the working folder
  ```sh
%cd /kaggle/working/bkai-igh-neopolyp-practice/
  ```
Test image process
  ```sh
!python3 infer.py --image_path 6f4d4987ea3b4bae5672a230194c5a08.jpeg
  ```
The resulting image will be stored in the "/kaggle/working/" folder. 

#Google drive link to the checkpoint 
https://drive.usercontent.google.com/download?id=1OZm_9OSPnvAy8GlEjpTgn9Ama0bEYpPS&authuser=0

The checkpoint is download and load automatically when infer.py is initialized so there is no need to place the checkpoint file in the working folder.
