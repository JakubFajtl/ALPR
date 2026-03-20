# License plate recognition template

The goal of the project is: given an input video you should recognize the license plates.  

## Input
You can see an example video under ``dataset/dummytestvideo.avi``.  
We recommend splitting the video into a training set and a testing set, and only using the testing part as a way to calculate your expected score.   
This is important because for grading we use a totally different video.

## Output
You can see an example of output under ``dataset/sampleOutput.csv``.  
Your output file should be in the same format.

## Project structure
The shell script ``evaluator.sh`` is used for running the project and calculating scores.  
This file initially runs your ``main.py`` file, followed by ``evaluation.py``. 
Do not modify either one of ``evaluator.sh`` or ``evaluation.py`` if you want to see proper outputs.

Rest of the project:
- ``CaptureFrame_Process.py`` for reading the input video
- ``Localization.py`` for figuring out the location of the plate in a frame
- ``Recognize.py`` for figuring out what characters are in a plate
- ``helpers/`` additional methods to help you get started (you do not have to use them, and are there for inspiration)
- ``requirements.txt`` if you want to use additional Python packages make sure to add them here as well
- ``.gitlab-ci.yml`` Gitlab pipeline file

## Pipeline
If you want to see your score you can change the file in ``.gitlab-ci.yml`` to run on the ``trainingvideo.avi``. 
We do **NOT** recommend always having this uncommented because running it on the full video makes the pipeline significantly slower.

## Tips for setting up

1. Python projects normally have a `requirements.txt` file with all the packages you need to install. To install the ones we currently picked for you, use:
    ```
    pip install -r requirements.txt
    ```
   Feel free to add more packages to the `requirements.txt` file, but make sure to ask your TA if you are allowed to use them!

2. It is good practice to use a Virtual Environment such that you can easily switch between Python projects (projects usually have different requirements, and some of them don't work well together). Some commonly used ones are 
    - `venv` (Built-in)  
    ```
    python -m venv myenv
    source myenv/bin/activate  # Linux/Mac
    myenv\Scripts\activate     # Windows
    ```
    - or `conda` (Anaconda Prompt)
    ```
    conda create --name myenv python=3.9
    conda activate myenv
    conda install <package_name_here>
    pip install -r requirements.txt
    ```

3. Help: `cv2 package not found`!   
Try installing `opencv-python`, so `pip install opencv-python`.

4. Help: `ffmpeg error` (even though we don't explicitly make use of ffmpeg, you might get this error when running your project because opencv uses it in the backend). Check if ffmpeg is installed with `ffmpeg -version`, if not, install it:
   - Windows: Download from [official website](https://ffmpeg.org/download.html) and add it to the PATH variable.
   - Linux: `sudo apt install ffmpeg`
   - Mac: `brew install ffmpeg`


