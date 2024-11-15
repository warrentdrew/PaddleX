---
comments: true
---

# Seal Recognition Pipeline Tutorial

## 1. Introduction to the Seal Recognition Pipeline
Seal recognition is a technology that automatically extracts and recognizes seal content from documents or images. The recognition of seal is part of document processing and has various applications in many scenarios, such as contract comparison, inventory access approval, and invoice reimbursement approval.

<img src="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/practical_tutorial/PP-ChatOCRv3_doc_seal/01.png">

The <b>Seal Recognition</b> pipeline includes a layout area analysis module, a seal detection module, and a text recognition module.

<b>If you prioritize model accuracy, please choose a model with higher accuracy. If you prioritize inference speed, please choose a model with faster inference. If you prioritize model storage size, please choose a model with a smaller storage footprint.</b>

<details><summary> 👉 Detailed Model List </summary>

<p><b>Layout Analysis Module Models:</b></p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet_layout_1x_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">Trained Model</a></td>
<td>86.8</td>
<td>13.0</td>
<td>91.3</td>
<td>7.4</td>
<td>An efficient layout area localization model trained on the PubLayNet dataset based on PicoDet-1x can locate five types of areas, including text, titles, tables, images, and lists.</td>
</tr>
<tr>
<td>PicoDet-S_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-S_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>87.1</td>
<td>13.5</td>
<td>45.8</td>
<td>4.8</td>
<td>An high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>70.3</td>
<td>13.6</td>
<td>46.2</td>
<td>4.8</td>
<td>A high-efficient layout area localization model trained on a self-constructed dataset based on PicoDet-S_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-L_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>89.3</td>
<td>15.7</td>
<td>159.8</td>
<td>22.6</td>
<td>An efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>79.9</td>
<td>17.2</td>
<td>160.2</td>
<td>22.6</td>
<td>A efficient layout area localization model trained on a self-constructed dataset based on PicoDet-L_layout_17cls for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/RT-DETR-H_layout_3cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">Trained Model</a></td>
<td>95.9</td>
<td>114.6</td>
<td>3832.6</td>
<td>470.1</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes three categories: tables, images, and seals.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Trained Model</a></td>
<td>92.6</td>
<td>115.1</td>
<td>3827.2</td>
<td>470.2</td>
<td>A high-precision layout area localization model trained on a self-constructed dataset based on RT-DETR-H for scenarios such as Chinese and English papers, magazines, and research reports includes 17 common layout categories, namely: paragraph titles, images, text, numbers, abstracts, content, chart titles, formulas, tables, table titles, references, document titles, footnotes, headers, algorithms, footers, and seals.</td>
</tr>
</tbody>
</table>
<p><b>Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built layout region analysis dataset, containing 10,000 images of common document types, including English and Chinese papers, magazines, research reports, etc. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Seal Detection Module Models</b>:</p>
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
<td>PP-OCRv4_server_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">Trained Model</a></td>
<td>98.21</td>
<td>84.341</td>
<td>2425.06</td>
<td>109</td>
<td>PP-OCRv4's server-side seal detection model, featuring higher accuracy, suitable for deployment on better-equipped servers</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_seal_det</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_seal_det_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">Trained Model</a></td>
<td>96.47</td>
<td>10.5878</td>
<td>131.813</td>
<td>4.6</td>
<td>PP-OCRv4's mobile seal detection model, offering higher efficiency, suitable for deployment on edge devices</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are evaluated on a self-built dataset containing 500 circular seal images. GPU inference time is based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Text Recognition Module Models</b>:</p>
<table>
<thead>
<tr>
<th>Model Name</th><th>Model Download Link</th>
<th>Average Recognition Accuracy (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time</th>
<th>Model Size (M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Trained Model</a></td>
<td>78.20</td>
<td>7.95018</td>
<td>46.7868</td>
<td>10.6 M</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Trained Model</a></td>
<td>79.20</td>
<td>7.19439</td>
<td>140.179</td>
<td>71.2 M</td>
</tr>
</tbody>
</table>
<p><b>Note: The evaluation set for the above accuracy indicators is a self-built Chinese dataset from PaddleOCR, covering various scenarios such as street scenes, web images, documents, and handwriting. The text recognition subset includes 11,000 images. The GPU inference time for all models above is based on an NVIDIA Tesla T4 machine with a precision type of FP32. The CPU inference speed is based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads, and the precision type is also FP32.</b></p></details>

## 2.  Quick Start
The pre trained model production line provided by PaddleX can quickly experience the effect. You can experience the effect of the seal recognition production line online, or use the command line or Python locally to experience the effect of the seal recognition production line.


Before using the seal recognition production line locally, please ensure that you have completed the wheel package installation of PaddleX according to the  [PaddleX Local Installation Guide](../../../installation/installation.en.md).

### 2.1 Command line experience
One command can quickly experience the effect of seal recognition production line, use [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png), and replace ` --input ` with the local path for prediction

```
paddlex --pipeline seal_recognition --input seal_text_det.png --device gpu:0 --save_path output
```

Parameter description:

```
--Pipeline: Production line name, here is the seal recognition production line
--Input: The local path or URL of the input image to be processed
--The GPU serial number used by the device (e.g. GPU: 0 indicates the use of the 0th GPU, GPU: 1,2 indicates the use of the 1st and 2nd GPUs), or the CPU (-- device CPU) can be selected for use
```

When executing the above Python script, the default seal recognition production line configuration file is loaded. If you need to customize the configuration file, you can execute the following command to obtain it:

<details><summary>  👉 Click to expand</summary>

<pre><code class="language-bash">paddlex --get_pipeline_config seal_recognition
</code></pre>
<p>After execution, the seal recognition production line configuration file will be saved in the current path. If you want to customize the save location, you can execute the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code class="language-bash">paddlex --get_pipeline_config seal_recognition --save_path ./my_path --save_path output
</code></pre>
<p>After obtaining the production line configuration file, you can replace '-- pipeline' with the configuration file save path to make the configuration file effective. For example, if the configuration file save path is <code>/ seal_recognition.yaml</code>， Just need to execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./seal_recognition.yaml --input seal_text_det.png --save_path output
</code></pre>
<p>Among them, parameters such as <code>--model</code> and <code>--device</code> do not need to be specified and will use the parameters in the configuration file. If the parameters are still specified, the specified parameters will prevail.</p></details>

After running, the result obtained is:

<details><summary>  👉 Click to expand</summary>

<pre><code>{'input_path': 'seal_text_det.png', 'layout_result': {'input_path': 'seal_text_det.png', 'boxes': [{'cls_id': 2, 'label': 'seal', 'score': 0.9813116192817688, 'coordinate': [0, 5.2238655, 639.59766, 637.6985]}]}, 'ocr_result': [{'input_path': PosixPath('/root/.paddlex/temp/tmp19fn93y5.png'), 'dt_polys': [array([[468, 469],
       [472, 469],
       [477, 471],
       [507, 501],
       [509, 505],
       [509, 509],
       [508, 513],
       [506, 514],
       [456, 553],
       [454, 555],
       [391, 581],
       [388, 581],
       [309, 590],
       [306, 590],
       [234, 577],
       [232, 577],
       [172, 548],
       [170, 546],
       [121, 504],
       [118, 501],
       [118, 496],
       [119, 492],
       [121, 490],
       [152, 463],
       [156, 461],
       [160, 461],
       [164, 463],
       [202, 495],
       [252, 518],
       [311, 530],
       [371, 522],
       [425, 501],
       [464, 471]]), array([[442, 439],
       [445, 442],
       [447, 447],
       [449, 490],
       [448, 494],
       [446, 497],
       [440, 499],
       [197, 500],
       [193, 499],
       [190, 496],
       [188, 491],
       [188, 448],
       [189, 444],
       [192, 441],
       [197, 439],
       [438, 438]]), array([[465, 341],
       [470, 344],
       [472, 346],
       [476, 356],
       [476, 419],
       [475, 424],
       [472, 428],
       [467, 431],
       [462, 433],
       [175, 434],
       [170, 433],
       [166, 430],
       [163, 426],
       [161, 420],
       [161, 354],
       [162, 349],
       [165, 345],
       [170, 342],
       [175, 340],
       [460, 340]]), array([[326,  34],
       [481,  85],
       [485,  88],
       [488,  90],
       [584, 220],
       [586, 225],
       [587, 229],
       [589, 378],
       [588, 383],
       [585, 388],
       [581, 391],
       [576, 393],
       [570, 392],
       [507, 373],
       [502, 371],
       [498, 367],
       [496, 359],
       [494, 255],
       [423, 162],
       [322, 129],
       [246, 151],
       [205, 169],
       [144, 252],
       [139, 360],
       [137, 365],
       [134, 369],
       [128, 373],
       [ 66, 391],
       [ 61, 392],
       [ 56, 390],
       [ 51, 387],
       [ 48, 382],
       [ 47, 377],
       [ 49, 230],
       [ 50, 225],
       [ 52, 221],
       [149,  89],
       [153,  86],
       [157,  84],
       [318,  34],
       [322,  33]])], 'dt_scores': [0.9943362380813267, 0.9994290391836306, 0.9945320407374245, 0.9908104427126033], 'rec_text': ['5263647368706', '吗繁物', '发票专用章', '天津君和缘商贸有限公司'], 'rec_score': [0.9921098351478577, 0.997374951839447, 0.9999369382858276, 0.9901710152626038]}]}
</code></pre></details>

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/seal_recognition/03.png">

The visualized image not saved by default. You can customize the save path through `--save_path`, and then all results will be saved in the specified path.


###  2.2 Python Script Integration
A few lines of code can complete the fast inference of the production line. Taking the seal recognition production line as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="seal_recognition")

output = pipeline.predict("seal_text_det.png")
for res in output:
    res.print()
    res.save_to_img("./output/") # Save the results in img
```

The result obtained is the same as the command line method.

In the above Python script, the following steps were executed:

（1）Instantiate the  production line object using `create_pipeline`: Specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the production line or the path to the production line configuration file. If it is the name of the production line, it must be supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for production line model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available if the production line supports it.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）Invoke the `predict` method of the  production line object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods, as shown in the following examples:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Parameter Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>Supports directly passing in Python variables, such as numpy.ndarray representing image data.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the path of the file to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in the URL of the file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png">Example</a>.</td>
</tr>
<tr>
<td>str</td>
<td>Supports passing in a local directory, which should contain files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td>dict</td>
<td>Supports passing in a dictionary type, where the key needs to correspond to a specific task, such as "img" for image classification tasks. The value of the dictionary supports the above types of data, for example: <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td>list</td>
<td>Supports passing in a list, where the list elements need to be of the above types of data, such as <code>[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
（3）Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list.

（4）Process the prediction results: The prediction result for each sample is of `dict` type and supports printing or saving to files, with the supported file types depending on the specific pipeline. For example:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Method Parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td>save_to_img</td>
<td>Save the results as an img format file</td>
<td><code>- save_path</code>: str, the path to save the file. When it's a directory, the saved file name will be consistent with the input file type;</td>
</tr>
</tbody>
</table>
Where `save_to_img` can save visualization results (including OCR result images, layout analysis result images).

If you have a configuration file, you can customize the configurations of the seal recognition  pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of the pipeline configuration file.

For example, if your configuration file is saved in `/ my_path/seal_recognition.yaml` ， Then only need to execute:


```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/seal_recognition.yaml")
output = pipeline.predict("seal_text_det.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存可视化结果
```

## 3. Development integration/deployment
If the production line can meet your requirements for inference speed and accuracy, you can directly develop integration/deployment.

If you need to directly apply the production line to your Python project, you can refer to the example code in [2.2.2 Python scripting] (# 222 python scripting integration).

In addition, PaddleX also offers three other deployment methods, detailed as follows:

🚀 ** High performance deployment: In actual production environments, many applications have strict standards for the performance indicators of deployment strategies, especially response speed, to ensure efficient system operation and smooth user experience. To this end, PaddleX provides a high-performance inference plugin aimed at deep performance optimization of model inference and pre-processing, achieving significant acceleration of end-to-end processes. For a detailed high-performance deployment process, please refer to the [PaddleX High Performance Deployment Guide] (../../../pipelin_deploy/high_performance_deploy. md).

☁️ ** Service deployment * *: Service deployment is a common form of deployment in actual production environments. By encapsulating inference functions as services, clients can access these services through network requests to obtain inference results. PaddleX supports users to achieve service-oriented deployment of production lines at low cost. For detailed service-oriented deployment processes, please refer to the PaddleX Service Deployment Guide (../../../ipeline_deploy/service_deploy. md).

Below are the API references and multi-language service invocation examples:

<details><summary>API Reference</summary>

<p>For main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>The request body and the response body are both JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the response body properties are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message. Fixed as <code>"Success"</code>.</td>
</tr>
</tbody>
</table>
<p>The response body may also have a <code>result</code> property of type <code>object</code>, which stores the operation result information.</p>
<ul>
<li>When the request is not processed successfully, the response body properties are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>Main operations provided by the service:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Obtain seal recognition results from an image.</p>
<p><code>POST /seal-recognition</code></p>
<ul>
<li>Request body properties:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The URL of an image file accessible by the service or the Base64 encoded result of the image file content.</td>
<td>Yes</td>
</tr>
<tr>
<td><code>inferenceParams</code></td>
<td><code>object</code></td>
<td>Inference parameters.</td>
<td>No</td>
</tr>
</tbody>
</table>
<p>Properties of <code>inferenceParams</code>:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>maxLongSide</code></td>
<td><code>integer</code></td>
<td>During inference, if the length of the longer side of the input image for the text detection model is greater than <code>maxLongSide</code>, the image will be scaled so that the length of the longer side equals <code>maxLongSide</code>.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> of the response body has the following properties:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>sealImpressions</code></td>
<td><code>array</code></td>
<td>Seal recognition results.</td>
</tr>
<tr>
<td><code>layoutImage</code></td>
<td><code>string</code></td>
<td>Layout area detection result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>sealImpressions</code> is an <code>object</code> with the following properties:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>poly</code></td>
<td><code>array</code></td>
<td>Text position. Elements in the array are the vertex coordinates of the polygon enclosing the text.</td>
</tr>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>Text content.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>Text recognition score.</td>
</tr>
</tbody>
</table></details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/seal-recognition&quot;
image_path = &quot;./demo.jpg&quot;
layout_image_path = &quot;./layout.jpg&quot;

with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(layout_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;layoutImage&quot;]))
print(f&quot;Output image saved at {layout_image_path}&quot;)
print(&quot;\nDetected seal impressions:&quot;)
print(result[&quot;sealImpressions&quot;])
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string imagePath = &quot;./demo.jpg&quot;;
    const std::string layoutImagePath = &quot;./layout.jpg&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector&lt;char&gt; buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr &lt;&lt; &quot;Error reading file.&quot; &lt;&lt; std::endl;
        return 1;
    }
    std::string bufferStr(reinterpret_cast&lt;const char*&gt;(buffer.data()), buffer.size());
    std::string encodedImage = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj[&quot;image&quot;] = encodedImage;
    std::string body = jsonObj.dump();

    auto response = client.Post(&quot;/seal-recognition&quot;, headers, body, &quot;application/json&quot;);
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;layoutImage&quot;];
        decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedLayoutImage(decodedString.begin(), decodedString.end());
        std::ofstream outputLayoutFile(layoutImagePath, std::ios::binary | std::ios::out);
        if (outputLayoutFile.is_open()) {
            outputLayoutFile.write(reinterpret_cast&lt;char*&gt;(decodedLayoutImage.data()), decodedLayoutImage.size());
            outputLayoutFile.close();
            std::cout &lt;&lt; &quot;Output image saved at &quot; &lt;&lt; layoutImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; layoutImagePath &lt;&lt; std::endl;
        }

        auto impressions = result[&quot;sealImpressions&quot;];
        std::cout &lt;&lt; &quot;\nDetected seal impressions:&quot; &lt;&lt; std::endl;
        for (const auto&amp; impression : impressions) {
            std::cout &lt;&lt; impression &lt;&lt; std::endl;
        }
    } else {
        std::cout &lt;&lt; &quot;Failed to send HTTP request.&quot; &lt;&lt; std::endl;
        return 1;
    }

    return 0;
}
</code></pre></details>

<details><summary>Java</summary>

<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = &quot;http://localhost:8080/seal-recognition&quot;;
        String imagePath = &quot;./demo.jpg&quot;;
        String layoutImagePath = &quot;./layout.jpg&quot;;

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;image&quot;, imageData);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get(&quot;result&quot;);
                String layoutBase64Image = result.get(&quot;layoutImage&quot;).asText();
                JsonNode impressions = result.get(&quot;sealImpressions&quot;);

                imageBytes = Base64.getDecoder().decode(layoutBase64Image);
                try (FileOutputStream fos = new FileOutputStream(layoutImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println(&quot;Output image saved at &quot; + layoutImagePath);

                System.out.println(&quot;\nDetected seal impressions: &quot; + impressions.toString());
            } else {
                System.err.println(&quot;Request failed with code: &quot; + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    &quot;bytes&quot;
    &quot;encoding/base64&quot;
    &quot;encoding/json&quot;
    &quot;fmt&quot;
    &quot;io/ioutil&quot;
    &quot;net/http&quot;
)

func main() {
    API_URL := &quot;http://localhost:8080/seal-recognition&quot;
    imagePath := &quot;./demo.jpg&quot;
    layoutImagePath := &quot;./layout.jpg&quot;

    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println(&quot;Error reading image file:&quot;, err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{&quot;image&quot;: imageData}
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

    client := &amp;http.Client{}
    req, err := http.NewRequest(&quot;POST&quot;, API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Println(&quot;Error creating request:&quot;, err)
        return
    }

    res, err := client.Do(req)
    if err != nil {
        fmt.Println(&quot;Error sending request:&quot;, err)
        return
    }
    defer res.Body.Close()

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println(&quot;Error reading response body:&quot;, err)
        return
    }
    type Response struct {
        Result struct {
            LayoutImage      string   `json:&quot;layoutImage&quot;`
            Impressions []map[string]interface{} `json:&quot;sealImpressions&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    layoutImageData, err := base64.StdEncoding.DecodeString(respData.Result.LayoutImage)
    if err != nil {
        fmt.Println(&quot;Error decoding base64 image data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(layoutImagePath, layoutImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing image to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.jpg\n&quot;, layoutImagePath)

    fmt.Println(&quot;\nDetected seal impressions:&quot;)
    for _, impression := range respData.Result.Impressions {
        fmt.Println(impression)
    }
}
</code></pre></details>

<details><summary>C#</summary>

<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = &quot;http://localhost:8080/seal-recognition&quot;;
    static readonly string imagePath = &quot;./demo.jpg&quot;;
    static readonly string layoutImagePath = &quot;./layout.jpg&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { &quot;image&quot;, image_data } };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string layoutBase64Image = jsonResponse[&quot;result&quot;][&quot;layoutImage&quot;].ToString();
        byte[] layoutImageBytes = Convert.FromBase64String(layoutBase64Image);
        File.WriteAllBytes(layoutImagePath, layoutImageBytes);
        Console.WriteLine($&quot;Output image saved at {layoutImagePath}&quot;);

        Console.WriteLine(&quot;\nDetected seal impressions:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;sealImpressions&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/seal-recognition'
const imagePath = './demo.jpg'
const layoutImagePath = &quot;./layout.jpg&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)
  })
};

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

axios.request(config)
.then((response) =&gt; {
    const result = response.data[&quot;result&quot;];

    imageBuffer = Buffer.from(result[&quot;layoutImage&quot;], 'base64');
    fs.writeFile(layoutImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${layoutImagePath}`);
    });

    console.log(&quot;\nDetected seal impressions:&quot;);
    console.log(result[&quot;sealImpressions&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/seal-recognition&quot;;
$image_path = &quot;./demo.jpg&quot;;
$layout_image_path = &quot;./layout.jpg&quot;;

$image_data = base64_encode(file_get_contents($image_path));
$payload = array(&quot;image&quot; =&gt; $image_data);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)[&quot;result&quot;];

file_put_contents($layout_image_path, base64_decode($result[&quot;layoutImage&quot;]));
echo &quot;Output image saved at &quot; . $layout_image_path . &quot;\n&quot;;

echo &quot;\nDetected seal impressions:\n&quot;;
print_r($result[&quot;sealImpressions&quot;]);

?&gt;
</code></pre></details>
</details>
<br/>

## 4.  Secondary development
If the default model weights provided by the seal recognition production line are not satisfactory in terms of accuracy or speed in your scenario, you can try using your own specific domain or application scenario data to further fine tune the existing model to improve the recognition performance of the seal recognition production line in your scenario.

### 4.1 Model fine-tuning
Due to the fact that the seal recognition production line consists of three modules, the performance of the model production line may not be as expected due to any of these modules.

You can analyze images with poor recognition performance and refer to the following rules for analysis and model fine-tuning:

* If the seal area is incorrectly located within the overall layout, the layout detection module may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/layout_detection.en.md#customization) section in the [Layout Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/layout_detection.en.md) and use your private dataset to fine-tune the layout detection model.
* If there is a significant amount of text that has not been detected (i.e. text miss detection phenomenon), it may be due to the shortcomings of the text detection model. You need to refer to the [Secondary Development](../../../module_usage/tutorials/ocr_modules/seal_text_detection.en.md#customization) section in the [Seal Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/seal_text_detection.en.md) to fine tune the text detection model using your private dataset.
* If seal texts are undetected (i.e., text miss detection), the text detection model may be insufficient. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/text_recognition.en.md#customization) section in the [Text Detection Module Development Tutorial](../../../module_usage/tutorials/ocr_modules/text_recognition.en.md) and use your private dataset to fine-tune the text detection model.

* If many detected texts contain recognition errors (i.e., the recognized text content does not match the actual text content), the text recognition model requires further improvement. You need to refer to the [Customization](../../../module_usage/tutorials/ocr_modules/text_recognition.en.md#customization) section.

### 4.2 Model Application
After completing fine-tuning training using a private dataset, you can obtain a local model weight file.

If you need to use the fine tuned model weights, simply modify the production line configuration file and replace the local path of the fine tuned model weights with the corresponding position in the production line configuration file

```python
......
 Pipeline:
  layout_model: RT-DETR-H_layout_3cls #can be modified to the local path of the fine tuned model
  text_det_model: PP-OCRv4_server_seal_det  #can be modified to the local path of the fine tuned model
  text_rec_model: PP-OCRv4_server_rec #can be modified to the local path of the fine tuned model
  layout_batch_size: 1
  text_rec_batch_size: 1
  device: "gpu:0"
......
```
Subsequently, refer to the command line or Python script in the local experience to load the modified production line configuration file.

##  5.  Multiple hardware support
PaddleX supports various mainstream hardware devices such as Nvidia GPU, Kunlun Core XPU, Ascend NPU, and Cambrian MLU, and can seamlessly switch between different hardware devices by simply modifying the <b>`--device`</b> parameter.

For example, if you use Nvidia GPU for inference on a seal recognition production line, the Python command you use is:

```
paddlex --pipeline seal_recognition --input seal_text_det.png --device gpu:0 --save_path output
```

At this point, if you want to switch the hardware to Ascend NPU, simply modify the ` --device ` in the Python command to NPU:

```
paddlex --pipeline seal_recognition --input seal_text_det.png --device npu:0 --save_path output
```

If you want to use the seal recognition production line on a wider range of hardware, please refer to the [PaddleX Multi Hardware Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md)。