---
comments: true
---

# Video Classification Module Development Tutorial

## I. Overview
The Video Classification Module is a crucial component in a computer vision system, responsible for categorizing input videos. The performance of this module directly impacts the accuracy and efficiency of the entire computer vision system. The Video Classification Module typically receives videos as input and then, through deep learning or other machine learning algorithms, classifies them into predefined categories based on their characteristics and content. For example, in an action recognition system, the Video Classification Module may need to classify input videos into categories such as "Abseiling," "Air Drumming," "Answering Questions," etc. The classification results of the Video Classification Module are output for use by other modules or systems.

## II. List of Supported Models


<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top1 Acc(%)</th>
<th>Model Storage Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PPTSM_ResNet50_k400_8frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PPTSM_ResNet50_k400_8frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PPTSM_ResNet50_k400_8frames_uniform_pretrained.pdparams">Trained Model</a></td>
<td>74.36</td>
<td>93.4 M</td>
<td rowspan="1">
PP-TSM is a video classification model developed by Baidu PaddlePaddle's Vision Team. This model is optimized based on the ResNet-50 backbone network and undergoes model tuning in six aspects: data augmentation, network structure fine-tuning, training strategies, Batch Normalization (BN) layer optimization, pre-trained model selection, and model distillation. Under the center crop evaluation method, its accuracy on Kinetics-400 is improved by 3.95 points compared to the original paper's implementation.
</td>
</tr>

<tr>
<td>PPTSMv2_LCNet_k400_8frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PPTSMv2_LCNet_k400_8frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PPTSMv2_LCNet_k400_8frames_uniform_pretrained.pdparams">Trained Model</a></td>
<td>71.71</td>
<td>22.5 M</td>
<td rowspan="2">PP-TSMv2 is a lightweight video classification model optimized based on the CPU-oriented model PP-LCNetV2. It undergoes model tuning in seven aspects: backbone network and pre-trained model selection, data augmentation, TSM module tuning, input frame number optimization, decoding speed optimization, DML distillation, and LTA module. Under the center crop evaluation method, it achieves an accuracy of 75.16%, with an inference speed of only 456ms on the CPU for a 10-second video input.</td>
</tr>
<tr>
<td>PPTSMv2_LCNet_k400_16frames_uniform</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PPTSMv2_LCNet_k400_16frames_uniform_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PPTSMv2_LCNet_k400_16frames_uniform_pretrained.pdparams">Trained Model</a></td>
<td>73.11</td>
<td>22.5 M</td>
</tr>

</table>

<p><b>Note: The above accuracy metrics refer to Top-1 Accuracy on the <a href="https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/dataset/k400.md">K400</a> validation set. </b><b>All model GPU inference times are based on NVIDIA Tesla T4 machines, with precision type FP32. CPU inference speeds are based on Intel® Xeon® Gold 5117 CPU @ 2.00GHz, with 8 threads and precision type FP32.</b></p></details>

## <span id="lable">III. Quick Integration</span>
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

After installing the wheel package, you can complete video classification module inference with just a few lines of code. You can switch between models in this module freely, and you can also integrate the model inference of the video classification module into your project. Before running the following code, please download the [demo video](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4) to your local machine.

```python
from paddlex import create_model
model = create_model("PPTSMv2_LCNet_k400_8frames_uniform")
output = model.predict("general_video_classification_001.mp4", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_video("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference APIs, please refer to the [PaddleX Single-Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you are seeking higher accuracy from existing models, you can use PaddleX's custom development capabilities to develop better video classification models. Before using PaddleX to develop video classification models, please ensure that you have installed the relevant model training plugins for video classification in PaddleX. The installation process can be found in the custom development section of the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare the dataset for the corresponding task module. PaddleX provides data validation functionality for each module, and <b>only data that passes data validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use your own private dataset for subsequent model training, please refer to the [PaddleX Video Classification Task Module Data Annotation Guide](../../../data_annotations/video_modules/video_classification.en.md).

#### 4.1.1 Demo Data Download
You can use the following command to download the demo dataset to a specified folder:
```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/k400_examples.tar -P ./dataset
tar -xf ./dataset/k400_examples.tar -C ./dataset/
```
#### 4.1.2 Data Validation
One command is all you need to complete data validation:

```bash
python main.py -c paddlex/configs/video_classification/PPTSMv2_LCNet_k400_8frames_uniform.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/k400_examples
```
After executing the above command, PaddleX will validate the dataset and summarize its basic information. If the command runs successfully, it will print `Check dataset passed !` in the log. The validation results file is saved in `./output/check_dataset_result.json`, and related outputs are saved in the `./output/check_dataset` directory in the current directory, including visual examples of sample images and sample distribution histograms.

<details><summary>👉 <b>Validation Results Details (Click to Expand)</b></summary>

<pre><code class="language-bash">{ "done_flag": true,
  "check_pass": true,
  "attributes": {
    "label_file": "../../dataset/k400_examples/label.txt",
    "num_classes": 5,
    "train_samples": 250,
    "train_sample_paths": [
      "check_dataset/../../dataset/k400_examples/videos/Wary2ON3aSo_000079_000089.mp4",
      "check_dataset/../../dataset/k400_examples/videos/_LHpfh0rXjk_000012_000022.mp4",
      "check_dataset/../../dataset/k400_examples/videos/dyoiNbn80q0_000039_000049.mp4",
      "check_dataset/../../dataset/k400_examples/videos/brBw6cFwock_000049_000059.mp4",
      "check_dataset/../../dataset/k400_examples/videos/-o4X5Z_Isyc_000085_000095.mp4",
      "check_dataset/../../dataset/k400_examples/videos/e24p-4W3TiU_000011_000021.mp4",
      "check_dataset/../../dataset/k400_examples/videos/2Grg_zwmYZE_000004_000014.mp4",
      "check_dataset/../../dataset/k400_examples/videos/aZY_0UqRNgA_000098_000108.mp4",
      "check_dataset/../../dataset/k400_examples/videos/WZlsi4nQHOo_000025_000035.mp4",
      "check_dataset/../../dataset/k400_examples/videos/rRh-lkFj4Tw_000001_000011.mp4"
    ],
    "val_samples": 50,
    "val_sample_paths": [
      "check_dataset/../../dataset/k400_examples/videos/7Mga5kywfU4.mp4",
      "check_dataset/../../dataset/k400_examples/videos/w5UCdQ2NmfY.mp4",
      "check_dataset/../../dataset/k400_examples/videos/Qbo_tnzfjOY.mp4",
      "check_dataset/../../dataset/k400_examples/videos/LgW8pMDtylE.mkv",
      "check_dataset/../../dataset/k400_examples/videos/BY0883Dvt1c.mp4",
      "check_dataset/../../dataset/k400_examples/videos/PHQkMPu-KNo.mp4",
      "check_dataset/../../dataset/k400_examples/videos/7LSJ2Ryv1a8.mp4",
      "check_dataset/../../dataset/k400_examples/videos/oBYZWvlI8Uk.mp4",
      "check_dataset/../../dataset/k400_examples/videos/dpn2eg9O3Rs.mkv",
      "check_dataset/../../dataset/k400_examples/videos/hXtsZAaZ3yc.mkv"
    ]
  },
  "analysis": {
    "histogram": "check_dataset/histogram.png"
  },
  "dataset_path": "./dataset/k400_examples",
  "show_type": "video",
  "dataset_type": "VideoClsDataset"
}
</code></pre>
<p>The above validation results, with check_pass being True, indicate that the dataset format meets the requirements. Explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>: The number of classes in this dataset is 5;</li>
<li><code>attributes.train_samples</code>: The number of training set samples in this dataset is 250;</li>
<li><code>attributes.val_samples</code>: The number of validation set samples in this dataset is 50;</li>
<li><code>attributes.train_sample_paths</code>: A list of relative paths to the visual samples in the training set of this dataset;</li>
<li><code>attributes.val_sample_paths</code>: A list of relative paths to the visual samples in the validation set of this dataset;</li>
</ul>
<p>Additionally, the dataset validation analyzes the sample number distribution across all classes in the dataset and generates a distribution histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/video_classification/01.png"></p></details>

#### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)
After completing data validation, you can convert the dataset format or re-split the training/validation ratio of the dataset by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Dataset Format Conversion/Dataset Splitting Details (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Image classification does not currently support data conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>The parameters for dataset splitting can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. The following are example explanations for some of the parameters in the configuration file:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. When set to <code>True</code>, the dataset format will be converted. The default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, you need to set the percentage of the training set, which should be an integer between 0-100, ensuring that the sum with <code>val_percent</code> equals 100;</li>
</ul>
<p>For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, you need to modify the configuration file as follows:</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 90
    val_percent: 10
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/video_classification/PPTSMv2_LCNet_k400_8frames_uniform.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/k400_examples
</code></pre>
<p>After the data splitting is executed, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>These parameters also support being set through appending command line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/video_classification/PPTSMv2_LCNet_k400_8frames_uniform.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/k400_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
A single command can complete the model training. Taking the training of the video classification model PPTSMv2_LCNet_k400_8frames_uniform as an example:
```
python main.py -c paddlex/configs/video_classification/PPTSMv2_LCNet_k400_8frames_uniform.yaml  \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/k400_examples
```

the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PPTSMv2_LCNet_k400_8frames_uniform.yaml`. When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the path of the training dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the fields under `Global` and `Train` in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first 2 GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the configuration file parameter instructions for the corresponding task module of the model [PaddleX Common Model Configuration File Parameters](../../instructions/config_parameters_common.en.md).


<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<ul>
<li>During model training, PaddleX automatically saves the model weight files, with the default being <code>output</code>. If you need to specify a save path, you can set it through the <code>-o Global.output</code> field in the configuration file.</li>
<li>PaddleX shields you from the concepts of dynamic graph weights and static graph weights. During model training, both dynamic and static graph weights are produced, and static graph weights are selected by default for model inference.</li>
<li>
<p>After completing the model training, all outputs are saved in the specified output directory (default is <code>./output/</code>), typically including:</p>
</li>
<li>
<p><code>train_result.json</code>: Training result record file, recording whether the training task was completed normally, as well as the output weight metrics, related file paths, etc.;</p>
</li>
<li><code>train.log</code>: Training log file, recording changes in model metrics and loss during training;</li>
<li><code>config.yaml</code>: Training configuration file, recording the hyperparameter configuration for this training session;</li>
<li><code>.pdparams</code>, <code>.pdema</code>, <code>.pdopt.pdstate</code>, <code>.pdiparams</code>, <code>.pdmodel</code>: Model weight-related files, including network parameters, optimizer, EMA, static graph network parameters, static graph network structure, etc.;</li>
</ul></details>

## <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model accuracy. Using PaddleX for model evaluation, a single command can complete the model evaluation:
```bash
python main.py -c  paddlex/configs/video_classification/PPTSMv2_LCNet_k400_8frames_uniform.yaml  \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/k400_examples
```
Similar to model training, the following steps are required:

* Specify the path of the model's `.yaml` configuration file (here it is `PPTSMv2_LCNet_k400_8frames_uniform.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path of the validation dataset: `-o Global.dataset_dir`. Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration. Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weight file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results. Specifically, it records whether the evaluation task was completed successfully and the model's evaluation metrics, including val.top1, val.top5;</p></details>

### <b>4.4 Model Inference and Model Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference prediction through the command line, simply use the following command. Before running the following code, please download the [demo video](https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4) to your local machine.

```bash
python main.py -c paddlex/configs/video_classification/PPTSMv2_LCNet_k400_8frames_uniform.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="general_video_classification_001.mp4"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PPTSMv2_LCNet_k400_8frames_uniform.yaml`)
* Specify the mode as model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Description](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipelines or directly into your own project.

1.<b>Pipeline Integration</b>


The video classification module can be integrated into the [General Video Classification Pipeline](../../../pipeline_usage/tutorials/video_pipelines/video_classification.en.md) of PaddleX. Simply replace the model path to update the video classification module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your obtained model.

2.<b>Module Integration</b>

The weights you produce can be directly integrated into the video classification module. You can refer to the Python example code in <a href="#lable">Quick Integration</a>  and simply replace the model with the path to your trained model.