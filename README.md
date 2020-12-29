# Useful Logs
Command to generate .xml and .bin for Mobile net ssd v2(Relative NXG DL system):
```
python3 mo_tf.py --input_model /home/nxgdl/HandModule/v2NUC_Hands_working/HandPoseNUCModule/hand_inference_graph/frozen_inference_graph.pb  --output_dir /home/nxgdl/HandModule/v2NUC_Hands_working/OpenVinoIR/HandIR/ --tensorflow_use_custom_operations_config /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json  --tensorflow_object_detection_api_pipeline_config /home/nxgdl/HandModule/v2NUC_Hands_working/HandPoseNUCModule/model-checkpoint/ssd_mobilenet_v1_coco.config  --input_shape [1,300,300,3] --data_type FP32
```
```
python3 /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_optimizer/mo_tf.py --input_model /home/nxgdl/HandModule/v2NUC_Hands_working/HandPoseNUCModule/hand_inference_graph/frozen_inference_graph.pb  --output_dir /home/nxgdl/HandModule/v2NUC_Hands_working/OpenVinoIR/HandIR/ --tensorflow_use_custom_operations_config /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json  --tensorflow_object_detection_api_pipeline_config /home/nxgdl/HandModule/v2NUC_Hands_working/HandPoseNUCModule/model-checkpoint/ssd_mobilenet_v1_coco.config  --input_shape [1,300,300,3] --data_type FP16
```

For cpu extension: type the command $ locate libcpu_extension.so	
```
python3 InferenceEngine_HandDetection.py -m /home/nxgdl/HandModule/v2NUC_Hands_working/OpenVinoIR/HandIR/frozen_inference_graph.xml -i cam -l /home/nxgdl/inference_engine_samples/intel64/Release/lib/libcpu_extension.so -d CPU --labels /home/nxgdl/HandModule/v2NUC_Hands_working/OpenVinoIR/HandIR/frozen_inference_graph.mapping
```


## To run on Intel Movidius/Compute Stick ..
##First generate the Intermediate representation to FP16 format;;

#using the command like::
'''
python3 /opt/intel/computer_vision_sdk_2018.3.343/deployment_tools/model_optimizer/mo_tf.py --input_model /home/nxgdl/HandModule/v2NUC_Hands_working/HandPoseNUCModule/hand_inference_graph/frozen_inference_graph.pb  --output_dir /home/nxgdl/HandModule/v2NUC_Hands_working/OpenVinoIR/HandIR/ --tensorflow_use_custom_operations_config /opt/intel/computer_vision_sdk/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json  --tensorflow_object_detection_api_pipeline_config /home/nxgdl/HandModule/v2NUC_Hands_working/HandPoseNUCModule/model-checkpoint/ssd_mobilenet_v1_coco.config  --input_shape [1,300,300,3] --data_type FP16

'''

##Now to run 

'''
 python3 InferenceEngine_HandDetection.py -m /home/nxgdl/HandModule/v2NUC_Hands_working/OpenVinoIR/HandIR/frozen_inference_graph.xml -i cam -l /home/nxgdl/inference_engine_samples/intel64/Release/lib/libcpu_extension.so -d MYRIAD --labels /home/nxgdl/HandModule/v2NUC_Hands_working/OpenVinoIR/HandIR/frozen_inference_graph.mapping
'''


This runs the demo on NCS 2


#original readme



# [How to train an object detection model easy for free](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/) | DLology Blog



## How to Run

Easy way: run [this Colab Notebook](https://colab.research.google.com/github/Tony607/object_detection_demo/blob/master/tensorflow_object_detection_training_colab.ipynb).

Alternatively, if you want to use your images instead of ones comes with this repo.

Require [Python 3.5+](https://www.python.org/ftp/python/3.6.4/python-3.6.4.exe) installed.
### Fork and clone this repository to your local machine.
```
https://github.com/Tony607/object_detection_demo
```
### Install required libraries
`pip3 install -r requirements.txt`


### Step 1: Annotate some images
- Save some photos with your custom object(s), ideally with `jpg` extension to `./data/raw` directory. (If your objects are simple like ones come with this repo, 20 images can be enough.)
- Resize those photo to uniformed size. e.g. `(800, 600)` with
```
python resize_images.py --raw-dir ./data/raw --save-dir ./data/images --ext jpg --target-size "(800, 600)"
```
Resized images locate in `./data/images/`
- Train/test split those files into two directories, `./data/images/train` and `./data/images/test`

- Annotate resized images with [labelImg](https://tzutalin.github.io/labelImg/), generate `xml` files inside `./data/images/train` and `./data/images/test` folders. 

*Tips: use shortcuts (`w`: draw box, `d`: next file, `a`: previous file, etc.) to accelerate the annotation.*

- Commit and push your annotated images and xml files (`./data/images/train` and `./data/images/test`) to your forked repository.


### Step 2: Open [Colab notebook](https://colab.research.google.com/github/Tony607/object_detection_demo/blob/master/tensorflow_object_detection_training_colab.ipynb)
- Replace the repository's url to yours and run it.


## How to run inference on frozen TensorFlow graph

Requirements:
- `frozen_inference_graph.pb` Frozen TensorFlow object detection model downloaded from Colab after training. 
- `label_map.pbtxt` File used to map correct name for predicted class index downloaded from Colab after training.

You can also opt to download my [copy](https://github.com/Tony607/object_detection_demo/releases/download/V0.1/checkpoint.zip) of those files from the GitHub Release page.


Run the following Jupyter notebook locally.
```
local_inference_test.ipynb
```
# [How to run TensorFlow object detection model faster with Intel Graphics](https://www.dlology.com/blog/how-to-run-tensorflow-object-detection-model-faster-with-intel-graphics/) | DLology Blog

## How to deploy the trained custom object detection model with OpenVINO

Requirements:
- Frozen TensorFlow object detection model. i.e. `frozen_inference_graph.pb` downloaded from Colab after training.
- The modified pipeline config file used for training. Also downloaded from Colab after training.

You can also opt to download my [copy](https://github.com/Tony607/object_detection_demo/releases/download/V0.1/checkpoint.zip) of those files from the GitHub Release page.

Run the following Jupyter notebook locally and follow the instructions in side.
```
deploy/openvino_convert_tf_object_detection.ipynb
```
## Run the benchmark

Examples

Benchmark SSD mobileNet V2 on GPU with FP16 quantized weights.
```
cd ./deploy
python openvino_inference_benchmark.py\
     --model-dir ./models/ssd_mobilenet_v2_custom_trained/FP16\
     --device GPU\
     --data-type FP16\
     --img ../test/15.jpg
```
TensorFlow benchmark on cpu
```
python local_inference_test.py\
     --model ./models/frozen_inference_graph.pb\
     --img ./test/15.jpg\
     --cpu
```
