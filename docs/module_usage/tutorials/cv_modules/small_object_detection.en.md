---
comments: true
---

# Small Object Detection Module Development Tutorial

## I. Overview
Small object detection typically refers to accurately detecting and locating small-sized target objects in images or videos. These objects often have a small pixel size in images, typically less than 32x32 pixels (as defined by datasets like MS COCO), and may be obscured by the background or other objects, making them difficult to observe directly by the human eye. Small object detection is an important research direction in computer vision, aiming to precisely detect small objects with minimal visual features in images.

## II. Supported Model List

<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5:0.95)</th>
<th>mAP(0.5)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description (VisDrone)</th>
</tr>
<tr>
<td>PP-YOLOE_plus_SOD-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE_plus_SOD-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-L_pretrained.pdparams">Trained Model</a></td>
<td>31.9</td>
<td>52.1</td>
<td>57.1</td>
<td>1007.0</td>
<td>324.93</td>
<td rowspan="3">PP-YOLOE_plus small object detection model trained on VisDrone. VisDrone is a benchmark dataset specifically for unmanned aerial vehicle (UAV) visual data, which is used for small object detection due to the small size of the targets and the inherent challenges they pose.</td>
</tr>
<tr>
<td>PP-YOLOE_plus_SOD-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE_plus_SOD-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-S_pretrained.pdparams">Trained Model</a></td>
<td>25.1</td>
<td>42.8</td>
<td>65.5</td>
<td>324.4</td>
<td>77.29</td>
</tr>
<tr>
<td>PP-YOLOE_plus_SOD-largesize-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE_plus_SOD-largesize-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-largesize-L_pretrained.pdparams">Trained Model</a></td>
<td>42.7</td>
<td>65.9</td>
<td>458.5</td>
<td>11172.7</td>
<td>340.42</td>
</tr>
</table>

<b>Note: The evaluation set for the above accuracy metrics is VisDrone-DET dataset mAP(0.5:0.95). GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b>


## III. Quick Integration  <a id="quick"> </a>
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md)

After installing the wheel package, you can complete the inference of the small object detection module with just a few lines of code. You can switch models under this module freely, and you can also integrate the model inference of the small object detection module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg) to your local machine.

```python
from paddlex import create_model

model_name = "PP-YOLOE_plus_SOD-S"

model = create_model(model_name)
output = model.predict("small_object_detection.jpg", batch_size=1)

for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference API, refer to the [PaddleX Single Model Python Script Usage Instructions](../../instructions/model_python_API.en.md).

## IV. Custom Development
If you seek higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better small object detection models. Before using PaddleX to develop small object detection models, ensure you have installed PaddleX's Detection-related model training capabilities. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare a dataset for the specific task module. PaddleX provides a data validation function for each module, and <b>only data that passes validation can be used for model training</b>. Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use a private dataset for model training, refer to [PaddleX Object Detection Task Module Data Annotation Tutorial](../../../data_annotations/cv_modules/object_detection.en.md).

#### 4.1.1 Demo Data Download
You can download the demo dataset to a specified folder using the following commands:

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/small_det_examples.tar -P ./dataset
tar -xf ./dataset/small_det_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation
You can complete data validation with a single command:

```bash
python main.py -c paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/small_det_examples
```
After executing the above command, PaddleX will validate the dataset and collect its basic information. Upon successful execution, the log will print the message `Check dataset passed !`. The validation result file will be saved in `./output/check_dataset_result.json`, and related outputs will be saved in the `./output/check_dataset` directory of the current directory. The output directory includes visualized example images and histograms of sample distributions.

<details><summary>👉 <b>Details of validation results (click to expand)</b></summary>

<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;num_classes&quot;: 10,
    &quot;train_samples&quot;: 1610,
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset/demo_img/9999938_00000_d_0000352.jpg&quot;,
      &quot;check_dataset/demo_img/9999941_00000_d_0000014.jpg&quot;,
      &quot;check_dataset/demo_img/9999973_00000_d_0000043.jpg&quot;
    ],
    &quot;val_samples&quot;: 548,
    &quot;val_sample_paths&quot;: [
      &quot;check_dataset/demo_img/0000330_00801_d_0000804.jpg&quot;,
      &quot;check_dataset/demo_img/0000103_00180_d_0000026.jpg&quot;,
      &quot;check_dataset/demo_img/0000291_04001_d_0000888.jpg&quot;
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/example_data/small_det_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;COCODetDataset&quot;
}
</code></pre>
<p>In the above validation results, <code>check_pass</code> being <code>True</code> indicates that the dataset format meets the requirements. The explanations for other indicators are as follows:</p>
<ul>
<li><code>attributes.num_classes</code>：The number of classes in this dataset is 10.</li>
<li><code>attributes.train_samples</code>：The number of samples in the training set of this dataset is 1610.</li>
<li><code>attributes.val_samples</code>：The number of samples in the validation set of this dataset is 548.</li>
<li><code>attributes.train_sample_paths</code>：A list of relative paths to the visualized images of samples in the training set of this dataset.</li>
<li><code>attributes.val_sample_paths</code>： A list of relative paths to the visualized images of samples in the validation set of this dataset.</li>
</ul>
<p>The dataset validation also analyzes the distribution of sample counts across all classes in the dataset and generates a histogram (histogram.png) to visualize this distribution. </p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/smallobj_det/01.png"></p></details>

#### 4.1.3 Dataset Format Conversion / Dataset Splitting (Optional)
After completing the dataset verification, you can convert the dataset format or re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details on Format Conversion / Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Small object detection supports converting datasets in <code>VOC</code> and <code>LabelMe</code> formats to <code>COCO</code> format.</p>
<p>Parameters related to dataset validation can be set by modifying the fields under <code>CheckDataset</code> in the configuration file. Examples of some parameters in the configuration file are as follows:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: Whether to perform dataset format conversion. Small object detection supports converting <code>VOC</code> and <code>LabelMe</code> format datasets to <code>COCO</code> format. Default is <code>False</code>;</li>
<li><code>src_dataset_type</code>: If dataset format conversion is performed, the source dataset format needs to be set. Default is <code>null</code>, with optional values <code>VOC</code>, <code>LabelMe</code>, <code>VOCWithUnlabeled</code>, <code>LabelMeWithUnlabeled</code>;
For example, if you want to convert a <code>LabelMe</code> format dataset to <code>COCO</code> format, taking the following <code>LabelMe</code> format dataset as an example, you need to modify the configuration as follows:</li>
</ul>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: LabelMe
  ......
</code></pre>
<p>Then execute the command:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./path/to/your_smallobject_labelme_dataset
</code></pre>
<p>Of course, the above parameters also support being set by appending command line arguments. Taking a <code>LabelMe</code> format dataset as an example:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./path/to/your_smallobject_labelme_dataset \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
</code></pre>
<p><b>(2) Dataset Splitting</b></p>
<p>Dataset splitting parameters can be set by modifying the <code>CheckDataset</code> section in the configuration file. Some example parameters in the configuration file are explained below:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. Set to <code>True</code> to enable dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set. The type is any integer between 0-100, ensuring the sum with <code>val_percent</code> is 100;</li>
</ul>
<p>For example, if you want to re-split the dataset with a 90% training set and a 10% validation set, modify the configuration file as follows:</p>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/small_det_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in their original paths.</p>
<p>The above parameters can also be set by appending command-line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-S.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/small_det_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
Model training can be completed with a single command, taking the training of `PP-YOLOE_plus_SOD-S` as an example:

```bash
python main.py -c paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-S.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/small_det_examples \
    -o Train.num_classes=10
```
The steps required are:

* Specify the `.yaml` configuration file path for the model (here it is `PP-YOLOE_plus_SOD-S.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Specify the mode as model training: `-o Global.mode=train`
* Specify the training dataset path: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file, or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters for Model Tasks](../../instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<ul>
<li>During model training, PaddleX automatically saves model weight files, defaulting to <code>output</code>. To specify a save path, use the <code>-o Global.output</code> field in the configuration file.</li>
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

### <b>4.3 Model Evaluation</b>
After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation, you can complete the evaluation with a single command:

```bash
python main.py -c paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-S.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/small_det_examples
```
Similar to model training, the process involves the following steps:

* Specify the path to the `.yaml` configuration file for the model（here it's `PP-YOLOE_plus_SOD-S.yaml`）
* Set the mode to model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`
Other related parameters can be configured by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For detailed information, please refer to [PaddleX Common Configuration Parameters for Models](../../instructions/config_parameters_common.en.md)。

<details><summary>👉 <b>More Details (Click to Expand)</b></summary>

<p>When evaluating the model, you need to specify the model weights file path. Each configuration file has a default weight save path built-in. If you need to change it, simply set it by appending a command line parameter, such as <code>-o Evaluate.weight_path=./output/best_model/best_model/model.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> file will be generated, which records the evaluation results, specifically whether the evaluation task was completed successfully, and the model's evaluation metrics, including AP.</p></details>

### <b>4.4 Model Inference</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions. In PaddleX, model inference predictions can be achieved through two methods: command line and wheel package.

#### 4.4.1 Model Inference
* To perform inference predictions through the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg) to your local machine.
```bash
python main.py -c paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-S.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="small_object_detection.jpg"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path for the model (here it is `PP-YOLOE_plus_SOD-S.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weight path: `-o Predict.model_dir="./output/best_model/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the `Global` and `Predict` fields in the `.yaml` configuration file. For details, please refer to [PaddleX Common Model Configuration File Parameter Explanation](../../instructions/config_parameters_common.en.md).

#### 4.4.2 Model Integration
The model can be directly integrated into the PaddleX pipelines or directly into your own projects.

1. <b>Pipeline Integration</b>

The small object detection module can be integrated into the [Small Object Detection Pipeline](../../../pipeline_usage/tutorials/cv_pipelines/small_object_detection.en.md) of PaddleX. Simply replace the model path to update the small object detection module of the relevant pipeline. In pipeline integration, you can use high-performance inference and service-oriented deployment to deploy your obtained model.

2. <b>Module Integration</b>

The weights you produce can be directly integrated into the small object detection module. You can refer to the Python example code in [Quick Integration](#quick), simply replacing the model with the path to your trained model.
