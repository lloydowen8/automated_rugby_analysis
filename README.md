# An Automated Territory and Possession Analysis of Broadcast Rugby Footage
## Repo Structure

After the setup instructions have been followed there should be 3 top-level directories: 

- Data: Evaluation & input data (TODO fix/remove notebooks and add README)
- Detectron2: Extrnal library used for object detection
- src: The source code for the project (TODO add README and tidy code)

Each has there own README describing the contents of the folder. 

## Setup Instructions
### Data folder
The data folder was to large to include with the original submission. Download the zipped folder from the following link: 

https://drive.google.com/drive/folders/1xQYL0U3YRcFe5GBrtmxFCtgxvOyQvQS0?usp=sharing

and extract the folder into the top-level directory of this repository.

### Windows
```
python -m venv .venv
.venv\Scripts\activate.bat
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2 
pip install -r requirements.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python src/demo/demo.py --config-file Data/config_files/config_eng_nzl.yaml
```

### Linux

For linux users replace
```
.venv\Scripts\activate.bat
```
with
```
source .venv/bin/activate
```

It can take a while to install all the dependencies (torch is around 4GB in size.)

The version of python used was 3.9.6, the code has not been tested on earlier versions. I'd also recommend running the code on a machine with a minimum of 8-12GB of RAM and a dedicated GPU. The default config has verbose set to true which will demonstrate the dictionary generation process and sports field registration for each frame, this can be disabled by setting the verbose option in `Data/config_files/config_eng_nzl.yaml` to False. 

If all else fails, there is an example video in the downloaded Data folder demonstrating the system. The accompanying video to this is the file located in Data/input_videos/eng_nzl_clip_less_frames.mp4
