---
comments: true
---

# Text Detection Module Development Tutorial

## I. Overview
The text detection module is a crucial component in OCR (Optical Character Recognition) systems, responsible for locating and marking regions containing text within images. The performance of this module directly impacts the accuracy and efficiency of the entire OCR system. The text detection module typically outputs bounding boxes for text regions, which are then passed on to the text recognition module for further processing.

## II. Supported Models
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_server_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">Trained Model</a></td>
<td>82.69</td>
<td>83.3501</td>
<td>2434.01</td>
<td>109</td>
<td>The server-side text detection model of PP-OCRv4, featuring higher accuracy and suitable for deployment on high-performance servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">Trained Model</a></td>
<td>77.79</td>
<td>10.6923</td>
<td>120.177</td>
<td>4.7</td>
<td>The mobile text detection model of PP-OCRv4, optimized for efficiency and suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>
## III. Quick Integration
> ❗ Before quick integration, please install the PaddleX wheel package. For detailed instructions, refer to the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

Just a few lines of code can complete the inference of the text detection module, allowing you to easily switch between models under this module. You can also integrate the model inference of the text detection module into your project. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png) to your local machine.

```python
from paddlex import create_model
model = create_model("PP-OCRv4_mobile_det")
output = model.predict("general_ocr_001.png", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
For more information on using PaddleX's single-model inference APIs, refer to the [PaddleX Single Model Python Script Usage Instructions](../../../module_usage/instructions/model_python_API.en.md).

## IV. Custom Development
If you seek even higher accuracy from existing models, you can leverage PaddleX's custom development capabilities to develop better text detection models. Before developing text detection models with PaddleX, ensure you have installed the PaddleOCR plugin for PaddleX. The installation process can be found in the [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 4.1 Data Preparation
Before model training, you need to prepare a dataset for the specific task module. PaddleX provides data validation functionality for each module, and <b>only data that passes validation can be used for model training</b>.
Additionally, PaddleX provides demo datasets for each module, which you can use to complete subsequent development. If you wish to use private datasets for model training, refer to the [PaddleX Text Detection/Text Recognition Task Module Data Annotation Tutorial](../../../data_annotations/ocr_modules/text_detection_recognition.en.md).

#### 4.1.1 Demo Data Download

You can use the following commands to download the demo dataset to a specified folder:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_det_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_det_dataset_examples.tar -C ./dataset/
```

#### 4.1.2 Data Validation

A single command can complete data validation:

```bash
python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples
```

After executing the above command, PaddleX will validate the dataset and gather basic information about it. Once the command runs successfully, `Check dataset passed !` will be printed in the log. The validation result file is saved in `./output/check_dataset_result.json`, and related outputs will be stored in the `./output/check_dataset` directory in the current directory. The output directory includes sample images and histograms of sample distribution.

<details><summary>👉 <b>Validation Result Details (Click to Expand)</b></summary>
<p>The specific content of the validation result file is:</p>
<pre><code class="language-bash">{
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_samples&quot;: 200,
    &quot;train_sample_paths&quot;: [
      &quot;../dataset/ocr_det_dataset_examples/images/train_img_61.jpg&quot;,
      &quot;../dataset/ocr_det_dataset_examples/images/train_img_289.jpg&quot;
    ],
    &quot;val_samples&quot;: 50,
    &quot;val_sample_paths&quot;: [
      &quot;../dataset/ocr_det_dataset_examples/images/val_img_61.jpg&quot;,
      &quot;../dataset/ocr_det_dataset_examples/images/val_img_137.jpg&quot;
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/ocr_det_dataset_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;TextDetDataset&quot;
}
</code></pre>
<p>In the above validation result, <code>check_pass</code> being <code>true</code> indicates that the dataset format meets the requirements. The explanation of other metrics is as follows:</p>
<ul>
<li><code>attributes.train_samples</code>: The number of training samples in the dataset is 200;</li>
<li><code>attributes.val_samples</code>: The number of validation samples in the dataset is 50;</li>
<li><code>attributes.train_sample_paths</code>: List of relative paths for visualizing training sample images in the dataset;</li>
<li><code>attributes.val_sample_paths</code>: List of relative paths for visualizing validation sample images in the dataset;</li>
</ul>
<p>Additionally, the dataset validation also analyzed the distribution of the length and width of all images in the dataset and plotted a distribution histogram (histogram.png):</p>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/text_det/01.png"></p></details>

### 4.1.3 Dataset Format Conversion/Dataset Splitting (Optional)

After completing data validation, you can convert the dataset format and re-split the training/validation ratio by <b>modifying the configuration file</b> or <b>appending hyperparameters</b>.

<details><summary>👉 <b>Details on Format Conversion/Dataset Splitting (Click to Expand)</b></summary>

<p><b>(1) Dataset Format Conversion</b></p>
<p>Text detection does not support data format conversion.</p>
<p><b>(2) Dataset Splitting</b></p>
<p>The parameters for dataset splitting can be set by modifying the <code>CheckDataset</code> section in the configuration file. Below are some example explanations for the parameters in the configuration file:</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: Whether to re-split the dataset. Set to <code>True</code> to enable dataset splitting, default is <code>False</code>;</li>
<li><code>train_percent</code>: If re-splitting the dataset, set the percentage of the training set. The type is any integer between 0-100, and the sum with <code>val_percent</code> must be 100;</li>
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
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples
</code></pre>
<p>After dataset splitting, the original annotation files will be renamed to <code>xxx.bak</code> in the original path.</p>
<p>The above parameters can also be set by appending command-line arguments:</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=90 \
    -o CheckDataset.split.val_percent=10
</code></pre></details>

### 4.2 Model Training
Model training can be completed with a single command. Here's an example of training the PP-OCRv4 mobile text detection model (`PP-OCRv4_mobile_det`):

```bash
python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples
```
The steps required are:

* Specify the path to the model's `.yaml` configuration file (here it's `PP-OCRv4_mobile_det.yaml`,When training other models, you need to specify the corresponding configuration files. The relationship between the model and configuration files can be found in the [PaddleX Model List (CPU/GPU)](../../../support_list/models_list.en.md))
* Set the mode to model training: `-o Global.mode=train`
* Specify the path to the training dataset: `-o Global.dataset_dir`
Other related parameters can be set by modifying the `Global` and `Train` fields in the `.yaml` configuration file or adjusted by appending parameters in the command line. For example, to specify training on the first two GPUs: `-o Global.device=gpu:0,1`; to set the number of training epochs to 10: `-o Train.epochs_iters=10`. For more modifiable parameters and their detailed explanations, refer to the [PaddleX Common Configuration Parameters Documentation](../../../module_usage/instructions/config_parameters_common.en.md).

<details><summary>👉 <b>More Information (Click to Expand)</b></summary>

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

### <b>4.3 Model Evaluation</b>

After completing model training, you can evaluate the specified model weight file on the validation set to verify the model's accuracy. Using PaddleX for model evaluation can be done with a single command:

```bash
python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ocr_det_dataset_examples
```

Similar to model training, the following steps are required:

* Specify the path to the model's `.yaml` configuration file (in this case, `PP-OCRv4_mobile_det.yaml`)
* Specify the mode as model evaluation: `-o Global.mode=evaluate`
* Specify the path to the validation dataset: `-o Global.dataset_dir`

Other related parameters can be set by modifying the fields under `Global` and `Evaluate` in the `.yaml` configuration file. For details, please refer to [PaddleX General Model Configuration File Parameter Instructions](../../../module_usage/instructions/config_parameters_common.md).

<details><summary>👉 <b>More Instructions (Click to Expand)</b></summary>

<p>During model evaluation, you need to specify the path to the model weight file. Each configuration file has a built-in default weight save path. If you need to change it, you can set it by adding a command line argument, such as <code>-o Evaluate.weight_path=./output/best_accuracy/best_accuracy.pdparams</code>.</p>
<p>After completing the model evaluation, an <code>evaluate_result.json</code> will be generated, which records the evaluation results. Specifically, it records whether the evaluation task was completed successfully and the model's evaluation metrics, including <code>precision</code>, <code>recall</code>, and <code>hmean</code>.</p></details>

### <b>4.4 Model Inference and Model Integration</b>
After completing model training and evaluation, you can use the trained model weights for inference predictions or Python integration.

#### 4.4.1 Model Inference
To perform inference predictions via the command line, simply use the following command. Before running the following code, please download the [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png) to your local machine.

```bash
python main.py -c paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_accuracy/inference" \
    -o Predict.input="general_ocr_001.png"
```
Similar to model training and evaluation, the following steps are required:

* Specify the `.yaml` configuration file path of the model (here it's `PP-OCRv4_mobile_det.yaml`)
* Set the mode to model inference prediction: `-o Global.mode=predict`
* Specify the model weights path: `-o Predict.model_dir="./output/best_accuracy/inference"`
* Specify the input data path: `-o Predict.input="..."`
Other related parameters can be set by modifying the fields under `Global` and `Predict` in the `.yaml` configuration file. For details, refer to [PaddleX Common Model Configuration File Parameter Description](../../../module_usage/instructions/config_parameters_common.en.md).

* Alternatively, you can use the PaddleX wheel package for inference, easily integrating the model into your own projects.

#### 4.4.2 Model Integration
Models can be directly integrated into PaddleX pipelines or into your own projects.

1.<b>Pipeline Integration</b>

The text detection module can be integrated into PaddleX pipelines such as the [General OCR Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/OCR.en.md), [Table Recognition Pipeline](../../../pipeline_usage/tutorials/ocr_pipelines/table_recognition.en.md), and [PP-ChatOCRv3-doc](../../../pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.en.md). Simply replace the model path to update the text detection module of the relevant pipeline.

2.<b>Module Integration</b>

The model weights you produce can be directly integrated into the text detection module. Refer to the Python example code in [Quick Integration](#iii-quick-integration), and simply replace the model with the path to your trained model.
