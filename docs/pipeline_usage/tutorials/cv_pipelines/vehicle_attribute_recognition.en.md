---
comments: true
---

# Vehicle Attribute Recognition Pipeline Tutorial

## 1. Introduction to Vehicle Attribute Recognition Pipeline
Vehicle attribute recognition is a crucial component in computer vision systems. Its primary task is to locate and label specific attributes of vehicles in images or videos, such as vehicle type, color, and license plate number. This task not only requires accurately detecting vehicles but also identifying detailed attribute information for each vehicle. The vehicle attribute recognition pipeline is an end-to-end serial system for locating and recognizing vehicle attributes, widely used in traffic management, intelligent parking, security surveillance, autonomous driving, and other fields. It significantly enhances system efficiency and intelligence levels, driving the development and innovation of related industries.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/vehicle_attribute_recognition/01.jpg">

<b>The vehicle attribute recognition pipeline includes a vehicle detection module and a vehicle attribute recognition module</b>, with several models in each module. Which models to use can be selected based on the benchmark data below. <b>If you prioritize model accuracy, choose models with higher accuracy; if you prioritize inference speed, choose models with faster inference; if you prioritize model storage size, choose models with smaller storage</b>.

<p><b>Vehicle Detection Module</b>:</p>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP 0.5:0.95</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-YOLOE-S_vehicle</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE-S_vehicle_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-S_vehicle_pretrained.pdparams">Trained Model</a></td>
<td>61.3</td>
<td>15.4</td>
<td>178.4</td>
<td>28.79</td>
<td rowspan="2">Vehicle detection model based on PP-YOLOE</td>
</tr>
<tr>
<td>PP-YOLOE-L_vehicle</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE-L_vehicle_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE-L_vehicle_pretrained.pdparams">Trained Model</a></td>
<td>63.9</td>
<td>32.6</td>
<td>775.6</td>
<td>196.02</td>
</tr>
</table>

<p><b>Note: The above accuracy metrics are mAP(0.5:0.95) on the PPVehicle validation set. All GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>
<p><b>Vehicle Attribute Recognition Module</b>:</p>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP (%)</th>
<th>GPU Inference Time (ms)</th>
<th>CPU Inference Time (ms)</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_vehicle_attribute</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-LCNet_x1_0_vehicle_attribute_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_vehicle_attribute_pretrained.pdparams">Trained Model</a></td>
<td>91.7</td>
<td>3.84845</td>
<td>9.23735</td>
<td>6.7 M</td>
<td>PP-LCNet_x1_0_vehicle_attribute is a lightweight vehicle attribute recognition model based on PP-LCNet.</td>
</tr>
</tbody>
</table>
<p><b>Note: The above accuracy metrics are mA on the VeRi dataset. GPU inference times are based on an NVIDIA Tesla T4 machine with FP32 precision. CPU inference speeds are based on an Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz with 8 threads and FP32 precision.</b></p>

## 2. Quick Start
The pre-trained models provided by PaddleX can quickly demonstrate results. You can experience the effects of the vehicle attribute recognition pipeline online or locally using command line or Python.

### 2.1 Online Experience
Not supported yet.

### 2.2 Local Experience
Before using the vehicle attribute recognition pipeline locally, ensure you have installed the PaddleX wheel package according to the [PaddleX Local Installation Tutorial](../../../installation/installation.en.md).

#### 2.2.1 Experience via Command Line
You can quickly experience the vehicle attribute recognition pipeline with a single command. Use the [test file](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_002.jpg) and replace `--input` with the local path for prediction.

```bash
paddlex --pipeline vehicle_attribute_recognition --input vehicle_attribute_002.jpg --device gpu:0
```
Parameter Description:

```
--pipeline: The name of the pipeline, here it is the vehicle attribute recognition pipeline.
--input: The local path or URL of the input image to be processed.
--device: The index of the GPU to use (e.g., gpu:0 means using the first GPU, gpu:1,2 means using the second and third GPUs). You can also choose to use the CPU (--device cpu).
```

When executing the above Python script, the default vehicle attribute recognition pipeline configuration file is loaded. If you need a custom configuration file, you can run the following command to obtain it:

<details><summary> 👉Click to Expand</summary>

<pre><code>paddlex --get_pipeline_config vehicle_attribute_recognition
</code></pre>
<p>After execution, the vehicle attribute recognition pipeline configuration file will be saved in the current directory. If you wish to specify a custom save location, you can run the following command (assuming the custom save location is <code>./my_path</code>):</p>
<pre><code>paddlex --get_pipeline_config vehicle_attribute_recognition --save_path ./my_path
</code></pre>
<p>After obtaining the pipeline configuration file, you can replace <code>--pipeline</code> with the saved path of the configuration file to make it effective. For example, if the saved path of the configuration file is <code>./vehicle_attribute_recognition.yaml</code>, just execute:</p>
<pre><code class="language-bash">paddlex --pipeline ./vehicle_attribute_recognition.yaml --input vehicle_attribute_002.jpg --device gpu:0
</code></pre>
<p>Among them, parameters such as <code>--model</code> and <code>--device</code> do not need to be specified, and the parameters in the configuration file will be used. If parameters are still specified, the specified parameters will take precedence.</p></details>

#### 2.2.2 Integrating via Python Script
A few lines of code suffice for rapid inference on the production line, taking the vehicle attribute recognition pipeline as an example:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="vehicle_attribute_recognition")

output = pipeline.predict("vehicle_attribute_002.jpg")
for res in output:
    res.print()  ## Print the structured output of the prediction
    res.save_to_img("./output/")  ## Save the visualized result image
    res.save_to_json("./output/")  ## Save the structured output of the prediction
```
The results obtained are the same as those from the command line method.

In the above Python script, the following steps are executed:

(1) Instantiate the `create_pipeline` to create a pipeline object: Specific parameter descriptions are as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>The name of the pipeline or the path to the pipeline configuration file. If it is the name of the pipeline, it must be a pipeline supported by PaddleX.</td>
<td><code>str</code></td>
<td>None</td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device for pipeline model inference. Supports: "gpu", "cpu".</td>
<td><code>str</code></td>
<td>"gpu"</td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>Whether to enable high-performance inference, only available when the pipeline supports high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
(2) Call the `predict` method of the vehicle attribute recognition pipeline object for inference prediction: The `predict` method parameter is `x`, which is used to input data to be predicted, supporting multiple input methods. Specific examples are as follows:

<table>
<thead>
<tr>
<th>Parameter Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>Supports directly passing in Python variables, such as image data represented by numpy.ndarray.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in the file path of the data to be predicted, such as the local path of an image file: <code>/root/data/img.jpg</code>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in the URL of the data file to be predicted, such as the network URL of an image file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_002.jpg">Example</a>.</td>
</tr>
<tr>
<td><code>str</code></td>
<td>Supports passing in a local directory, which should contain data files to be predicted, such as the local path: <code>/root/data/</code>.</td>
</tr>
<tr>
<td><code>dict</code></td>
<td>Supports passing in a dictionary type, where the key needs to correspond to the specific task, such as "img" for the vehicle attribute recognition task, and the value of the dictionary supports the above data types, for example: <code>{"img": "/root/data1"}</code>.</td>
</tr>
<tr>
<td><code>list</code></td>
<td>Supports passing in a list, where the elements of the list need to be the above data types, such as <code>[numpy.ndarray, numpy.ndarray], ["/root/data/img1.jpg", "/root/data/img2.jpg"], ["/root/data1", "/root/data2"], [{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}]</code>.</td>
</tr>
</tbody>
</table>
(3) Obtain the prediction results by calling the `predict` method: The `predict` method is a `generator`, so prediction results need to be obtained through iteration. The `predict` method predicts data in batches, so the prediction results are in the form of a list representing a set of prediction results.

(4) Processing the Prediction Results: The prediction result for each sample is in `dict` format, which supports printing or saving to a file. The supported file types for saving depend on the specific pipeline, such as:

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
<td>print</td>
<td>Print results to the terminal</td>
<td><code>- format_json</code>: bool, whether to format the output content with json indentation, default is True;<br/><code>- indent</code>: int, json formatting setting, only effective when format_json is True, default is 4;<br/><code>- ensure_ascii</code>: bool, json formatting setting, only effective when format_json is True, default is False;</td>
</tr>
<tr>
<td>save_to_json</td>
<td>Save results as a json file</td>
<td><code>- save_path</code>: str, the path to save the file, when it is a directory, the saved file name is consistent with the input file type;<br/><code>- indent</code>: int, json formatting setting, default is 4;<br/><code>- ensure_ascii</code>: bool, json formatting setting, default is False;</td>
</tr>
<tr>
<td>save_to_img</td>
<td>Save results as an image file</td>
<td><code>- save_path</code>: str, the path to save the file, when it is a directory, the saved file name is consistent with the input file type;</td>
</tr>
</tbody>
</table>
If you have obtained the configuration file, you can customize the configurations for the vehicle attribute recognition pipeline by simply modifying the `pipeline` parameter in the `create_pipeline` method to the path of your pipeline configuration file.

For example, if your configuration file is saved at `./my_path/vehicle_attribute_recognition.yaml`, you only need to execute:

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/vehicle_attribute_recognition.yaml")
output = pipeline.predict("vehicle_attribute_002.jpg")
for res in output:
    res.print()  # Print the structured output of the prediction
    res.save_to_img("./output/")  # Save the visualized result image
    res.save_to_json("./output/")  # Save the structured output of the prediction
```

## 3. Development Integration/Deployment
If the vehicle attribute recognition pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to directly apply the vehicle attribute recognition pipeline in your Python project, you can refer to the example code in [2.2.2 Python Script Integration](#222-python-script-integration).

Additionally, PaddleX provides three other deployment methods, detailed as follows:

🚀 <b>High-Performance Inference</b>: In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experience. To this end, PaddleX provides high-performance inference plugins aimed at deeply optimizing model inference and pre/post-processing to significantly speed up the end-to-end process. For detailed high-performance inference procedures, please refer to the [PaddleX High-Performance Inference Guide](../../../pipeline_deploy/high_performance_inference.md).

☁️ <b>Serving</b>: Serving is a common deployment strategy in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various solutions for serving pipelines. For detailed pipeline serving procedures, please refer to the [PaddleX Pipeline Serving Guide](../../../pipeline_deploy/serving.md).

Below are the API reference and multi-language service invocation examples for the basic serving solution:

<details><summary>API Reference</summary>

<p>For primary operations provided by the service:</p>
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
<td><code>logId</code></td>
<td><code>string</code></td>
<td>UUID for the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
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
<td><code>logId</code></td>
<td><code>string</code></td>
<td>UUID for the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>Primary operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Get vehicle attribute recognition results.</p>
<p><code>POST /vehicle-attribute-recognition</code></p>
<ul>
<li>The request body properties are as follows:</li>
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
<td>The URL of an image file accessible by the server or the Base64 encoded result of the image file content.</td>
<td>Yes</td>
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
<td><code>vehicles</code></td>
<td><code>array</code></td>
<td>Information about the vehicle's location and attributes.</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>The vehicle attribute recognition result image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>vehicles</code> is an <code>object</code> with the following properties:</p>
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
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>The location of the vehicle. The elements in the array are the x-coordinate of the top-left corner, the y-coordinate of the top-left corner, the x-coordinate of the bottom-right corner, and the y-coordinate of the bottom-right corner of the bounding box, respectively.</td>
</tr>
<tr>
<td><code>attributes</code></td>
<td><code>array</code></td>
<td>The vehicle attributes.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The detection score.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>attributes</code> is an <code>object</code> with the following properties:</p>
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
<td><code>label</code></td>
<td><code>string</code></td>
<td>The label of the attribute.</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>The classification score.</td>
</tr>
</tbody>
</table>
</details>

<details><summary>Multi-Language Service Invocation Examples</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/vehicle-attribute-recognition&quot;
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;\nDetected vehicles:&quot;)
print(result[&quot;vehicles&quot;])
</code></pre></details>
</details>
<br/>

📱 <b>Edge Deployment</b>: Edge deployment is a method where computing and data processing functions are placed on the user's device itself, allowing the device to process data directly without relying on remote servers. PaddleX supports deploying models on edge devices such as Android. For detailed edge deployment procedures, please refer to the [PaddleX Edge Deployment Guide](../../../pipeline_deploy/edge_deploy.en.md).
You can choose an appropriate method to deploy your model pipeline based on your needs, and proceed with subsequent AI application integration.


## 4. Custom Development
If the default model weights provided by the vehicle attribute recognition pipeline do not meet your expectations in terms of accuracy or speed for your specific scenario, you can try to further <b>fine-tune</b> the existing models using <b>your own data from specific domains or application scenarios</b> to enhance the recognition performance of the vehicle attribute recognition pipeline in your context.

### 4.1 Model Fine-tuning
Since the vehicle attribute recognition pipeline includes both a vehicle attribute recognition module and a vehicle detection module, the suboptimal performance of the pipeline may stem from either module.
You can analyze images with poor recognition results. If during the analysis, you find that many main targets are not detected, it may indicate deficiencies in the vehicle detection model. In this case, you need to refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/human_detection.en.md#4-custom-development) section in the [Vehicle Detection Module Development Tutorial](../../../module_usage/tutorials/cv_modules/human_detection.en.md) and use your private dataset to fine-tune the vehicle detection model. If the detected main attributes are incorrectly recognized, you need to refer to the [Custom Development](../../../module_usage/tutorials/cv_modules/vehicle_attribute_recognition.en.md#4-custom-development) section in the [Vehicle Attribute Recognition Module Development Tutorial](../../../module_usage/tutorials/cv_modules/vehicle_attribute_recognition.en.md) and use your private dataset to fine-tune the vehicle attribute recognition model.

### 4.2 Model Application
After fine-tuning with your private dataset, you will obtain local model weight files.

To use the fine-tuned model weights, you only need to modify the pipeline configuration file by replacing the path to the default model weights with the local path to the fine-tuned model weights:

```
......
Pipeline:
  det_model: PP-YOLOE-L_vehicle
  cls_model: PP-LCNet_x1_0_vehicle_attribute   # Can be modified to the local path of the fine-tuned model
  device: "gpu"
  batch_size: 1
......
```
Subsequently, refer to the command-line or Python script methods in the local experience, and load the modified pipeline configuration file.

## 5. Multi-hardware Support
PaddleX supports various mainstream hardware devices such as NVIDIA GPUs, Kunlun XPU, Ascend NPU, and Cambricon MLU. <b>Simply modifying the `--device` parameter</b> allows seamless switching between different hardware.

For example, if you use an NVIDIA GPU for inference with the vehicle attribute recognition pipeline, the command is:

```bash
paddlex --pipeline vehicle_attribute_recognition --input vehicle_attribute_002.jpg --device gpu:0
```
At this point, if you want to switch the hardware to Ascend NPU, you only need to change `--device` to npu:0:

```bash
paddlex --pipeline vehicle_attribute_recognition --input vehicle_attribute_002.jpg --device npu:0
```
If you want to use the vehicle attribute recognition pipeline on more types of hardware, please refer to the [PaddleX Multi-device Usage Guide](../../../other_devices_support/multi_devices_use_guide.en.md).
