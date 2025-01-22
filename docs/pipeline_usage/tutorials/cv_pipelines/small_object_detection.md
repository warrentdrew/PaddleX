---
comments: true
---

# 小目标检测产线使用教程

## 1. 小目标检测产线介绍
小目标检测是一种专门识别图像中体积较小物体的技术，广泛应用于监控、无人驾驶和卫星图像分析等领域。它能够从复杂场景中准确找到并分类像行人、交通标志或小动物等小尺寸物体。通过使用深度学习算法和优化的卷积神经网络，小目标检测可以有效提升对小物体的识别能力，确保在实际应用中不遗漏重要信息。这项技术在提高安全性和自动化水平方面发挥着重要作用。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/small_object_detection/01.png">

<b>小目标检测产线中包含了小目标检测模块，如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型</b>。


<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>mAP（%）</th>
<th>GPU推理耗时（ms）</th>
<th>CPU推理耗时（ms）</th>
<th>模型存储大小（M)</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-YOLOE_plus_SOD-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE_plus_SOD-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-S_pretrained.pdparams">训练模型</a></td>
<td>25.1</td>
<td>65.4608</td>
<td>324.37</td>
<td>77.3 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus_SOD-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE_plus_SOD-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-L_pretrained.pdparams">训练模型</a></td>
<td>31.9</td>
<td>57.1448</td>
<td>1006.98</td>
<td>325.0 M</td>
</tr>
<tr>
<td>PP-YOLOE_plus_SOD-largesize-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE_plus_SOD-largesize-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus_SOD-largesize-L_pretrained.pdparams">训练模型</a></td>
<td>42.7</td>
<td>458.521</td>
<td>11172.7</td>
<td>340.5 M</td>
</tr>
</tbody>
</table>
<p><b>注：以上精度指标为 </b><a href="https://github.com/VisDrone/VisDrone-Dataset">VisDrone-DET</a><b> 验证集 mAP(0.5:0.95)。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b></p>

## 2. 快速开始
PaddleX 支持在本地使用命令行或 Python 体验小目标检测产线的效果。

在本地使用小目标检测产线前，请确保您已经按照[PaddleX本地安装教程](../../../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.1 命令行方式体验
一行命令即可快速体验小目标检测产线效果，使用 [测试文件](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg)，并将 `--input` 替换为本地路径，进行预测

```bash
paddlex --pipeline small_object_detection --input small_object_detection.jpg --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为小目标检测产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```

在执行上述 Python 脚本时，加载的是默认的小目标检测产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details><summary> 👉点击展开</summary>

<pre><code>paddlex --get_pipeline_config small_object_detection
</code></pre>
<p>执行后，小目标检测产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为 <code>./my_path</code> ）：</p>
<pre><code>paddlex --get_pipeline_config small_object_detection --save_path ./my_path
</code></pre>
<p>获取产线配置文件后，可将 <code>--pipeline</code> 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 <code>./small_object_detection.yaml</code>，只需执行：</p>
<pre><code class="language-bash">paddlex --pipeline ./small_object_detection.yaml --input small_object_detection.jpg --device gpu:0
</code></pre>
<p>其中，<code>--model</code>、<code>--device</code> 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。</p></details>

运行后，得到的结果为：

```
{'input_path': 'small_object_detection.jpg', 'boxes': [{'cls_id': 3, 'label': 'car', 'score': 0.9243856072425842, 'coordinate': [624, 638, 682, 741]}, {'cls_id': 3, 'label': 'car', 'score': 0.9206348061561584, 'coordinate': [242, 561, 356, 613]}, {'cls_id': 3, 'label': 'car', 'score': 0.9194547533988953, 'coordinate': [670, 367, 705, 400]}, {'cls_id': 3, 'label': 'car', 'score': 0.9162291288375854, 'coordinate': [459, 259, 523, 283]}, {'cls_id': 4, 'label': 'van', 'score': 0.9075379371643066, 'coordinate': [467, 213, 498, 242]}, {'cls_id': 4, 'label': 'van', 'score': 0.9066920876502991, 'coordinate': [547, 351, 577, 397]}, {'cls_id': 3, 'label': 'car', 'score': 0.9041045308113098, 'coordinate': [502, 632, 562, 736]}, {'cls_id': 3, 'label': 'car', 'score': 0.8934890627861023, 'coordinate': [613, 383, 647, 427]}, {'cls_id': 3, 'label': 'car', 'score': 0.8803309202194214, 'coordinate': [640, 280, 671, 309]}, {'cls_id': 3, 'label': 'car', 'score': 0.8727016448974609, 'coordinate': [1199, 256, 1259, 281]}, {'cls_id': 3, 'label': 'car', 'score': 0.8705748915672302, 'coordinate': [534, 410, 570, 461]}, {'cls_id': 3, 'label': 'car', 'score': 0.8654043078422546, 'coordinate': [669, 248, 702, 271]}, {'cls_id': 3, 'label': 'car', 'score': 0.8555219769477844, 'coordinate': [525, 243, 550, 270]}, {'cls_id': 3, 'label': 'car', 'score': 0.8522038459777832, 'coordinate': [526, 220, 553, 243]}, {'cls_id': 3, 'label': 'car', 'score': 0.8392605185508728, 'coordinate': [557, 141, 575, 158]}, {'cls_id': 3, 'label': 'car', 'score': 0.8353804349899292, 'coordinate': [537, 120, 553, 133]}, {'cls_id': 3, 'label': 'car', 'score': 0.8322211503982544, 'coordinate': [585, 132, 603, 147]}, {'cls_id': 3, 'label': 'car', 'score': 0.8298957943916321, 'coordinate': [701, 283, 736, 313]}, {'cls_id': 3, 'label': 'car', 'score': 0.8217393159866333, 'coordinate': [885, 347, 943, 377]}, {'cls_id': 3, 'label': 'car', 'score': 0.820313572883606, 'coordinate': [493, 150, 511, 168]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.8183429837226868, 'coordinate': [203, 701, 224, 743]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.815082848072052, 'coordinate': [185, 710, 201, 744]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.7892289757728577, 'coordinate': [311, 371, 344, 407]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.7812919020652771, 'coordinate': [345, 380, 388, 405]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7748346328735352, 'coordinate': [295, 500, 309, 532]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.7688500285148621, 'coordinate': [851, 436, 863, 466]}, {'cls_id': 3, 'label': 'car', 'score': 0.7466475367546082, 'coordinate': [565, 114, 580, 128]}, {'cls_id': 3, 'label': 'car', 'score': 0.7156463265419006, 'coordinate': [483, 66, 495, 78]}, {'cls_id': 3, 'label': 'car', 'score': 0.704211950302124, 'coordinate': [607, 138, 642, 152]}, {'cls_id': 3, 'label': 'car', 'score': 0.7021926045417786, 'coordinate': [505, 72, 518, 83]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.6897469162940979, 'coordinate': [802, 460, 815, 488]}, {'cls_id': 3, 'label': 'car', 'score': 0.671891450881958, 'coordinate': [574, 123, 593, 136]}, {'cls_id': 9, 'label': 'motorcycle', 'score': 0.6712754368782043, 'coordinate': [445, 317, 472, 334]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.6695684790611267, 'coordinate': [479, 309, 489, 332]}, {'cls_id': 3, 'label': 'car', 'score': 0.6273623704910278, 'coordinate': [654, 210, 677, 234]}, {'cls_id': 3, 'label': 'car', 'score': 0.6070230603218079, 'coordinate': [640, 166, 667, 185]}, {'cls_id': 3, 'label': 'car', 'score': 0.6064521670341492, 'coordinate': [461, 59, 476, 71]}, {'cls_id': 3, 'label': 'car', 'score': 0.5860581398010254, 'coordinate': [464, 87, 484, 100]}, {'cls_id': 9, 'label': 'motorcycle', 'score': 0.5792551636695862, 'coordinate': [390, 390, 419, 408]}, {'cls_id': 3, 'label': 'car', 'score': 0.5559225678443909, 'coordinate': [481, 125, 496, 140]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.5531904697418213, 'coordinate': [869, 306, 880, 331]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.5468509793281555, 'coordinate': [895, 294, 904, 319]}, {'cls_id': 3, 'label': 'car', 'score': 0.5451828241348267, 'coordinate': [505, 94, 518, 108]}, {'cls_id': 3, 'label': 'car', 'score': 0.5398445725440979, 'coordinate': [657, 188, 681, 208]}, {'cls_id': 4, 'label': 'van', 'score': 0.5318890810012817, 'coordinate': [518, 88, 534, 102]}, {'cls_id': 3, 'label': 'car', 'score': 0.5296525359153748, 'coordinate': [527, 71, 540, 81]}, {'cls_id': 6, 'label': 'tricycle', 'score': 0.5168400406837463, 'coordinate': [528, 320, 563, 346]}, {'cls_id': 3, 'label': 'car', 'score': 0.5088561177253723, 'coordinate': [511, 84, 530, 95]}, {'cls_id': 0, 'label': 'pedestrian', 'score': 0.502006471157074, 'coordinate': [841, 266, 850, 283]}]}
```

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/small_object_detection/02.png">

可视化图片默认不进行保存，您可以通过 `--save_path` 自定义保存路径，随后所有结果将被保存在指定路径下。

### 2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，以通用小目标检测产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="small_object_detection")

output = pipeline.predict("small_object_detection.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化产线对象：具体参数说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline</code></td>
<td>产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。</td>
<td><code>str</code></td>
<td>无</td>
</tr>
<tr>
<td><code>device</code></td>
<td>产线模型推理设备。支持：“gpu”，“cpu”。</td>
<td><code>str</code></td>
<td><code>gpu</code></td>
</tr>
<tr>
<td><code>use_hpip</code></td>
<td>是否启用高性能推理，仅当该产线支持高性能推理时可用。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
</tbody>
</table>
（2）调用产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

<table>
<thead>
<tr>
<th>参数类型</th>
<th>参数说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>Python Var</td>
<td>支持直接传入Python变量，如numpy.ndarray表示的图像数据。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入待预测数据文件路径，如图像文件的本地路径：<code>/root/data/img.jpg</code>。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入待预测数据文件URL，如图像文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg">示例</a>。</td>
</tr>
<tr>
<td>str</td>
<td>支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code>。</td>
</tr>
<tr>
<td>dict</td>
<td>支持传入字典类型，字典的key需与具体任务对应，如图像分类任务对应\"img\"，字典的val支持上述类型数据，例如：<code>{\"img\": \"/root/data1\"}</code>。</td>
</tr>
<tr>
<td>list</td>
<td>支持传入列表，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]，[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code>，<code>[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]</code>。</td>
</tr>
</tbody>
</table>
（3）调用`predict`方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果。

（4）对预测结果进行处理：每个样本的预测结果均为`dict`类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

<table>
<thead>
<tr>
<th>方法</th>
<th>说明</th>
<th>方法参数</th>
</tr>
</thead>
<tbody>
<tr>
<td>print</td>
<td>打印结果到终端</td>
<td><code>- format_json</code>：bool类型，是否对输出内容进行使用json缩进格式化，默认为True；<br/><code>- indent</code>：int类型，json格式化设置，仅当format_json为True时有效，默认为4；<br/><code>- ensure_ascii</code>：bool类型，json格式化设置，仅当format_json为True时有效，默认为False；</td>
</tr>
<tr>
<td>save_to_json</td>
<td>将结果保存为json格式的文件</td>
<td><code>- save_path</code>：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；<br/><code>- indent</code>：int类型，json格式化设置，默认为4；<br/><code>- ensure_ascii</code>：bool类型，json格式化设置，默认为False；</td>
</tr>
<tr>
<td>save_to_img</td>
<td>将结果保存为图像格式的文件</td>
<td><code>- save_path</code>：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；</td>
</tr>
</tbody>
</table>
若您获取了配置文件，即可对小目标检测产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/small_object_detection.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/small_object_detection.yaml")
output = pipeline.predict("small_object_detection.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_img("./output/") ## 保存结果可视化图像
    res.save_to_json("./output/") ## 保存预测的结构化输出
```

## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 <b>高性能推理</b>：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能推理流程请参考[PaddleX高性能推理指南](../../../pipeline_deploy/high_performance_inference.md)。

☁️ <b>服务化部署</b>：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持多种产线服务化部署方案，详细的产线服务化部署流程请参考[PaddleX服务化部署指南](../../../pipeline_deploy/serving.md)。

以下是基础服务化部署的API参考与多语言服务调用示例：

<details><summary>API参考</summary>

<p>对于服务提供的主要操作：</p>
<ul>
<li>HTTP请求方法为POST。</li>
<li>请求体和响应体均为JSON数据（JSON对象）。</li>
<li>当请求处理成功时，响应状态码为<code>200</code>，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。固定为<code>0</code>。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。固定为<code>"Success"</code>。</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>操作结果。</td>
</tr>
</tbody>
</table>
<ul>
<li>当请求处理未成功时，响应体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>请求的UUID。</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>错误码。与响应状态码相同。</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>错误说明。</td>
</tr>
</tbody>
</table>
<p>服务提供的主要操作如下：</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>对图像进行目标检测。</p>
<p><code>POST /small-object-detection</code></p>
<ul>
<li>请求体的属性如下：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
<th>是否必填</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>服务器可访问的图像文件的URL或图像文件内容的Base64编码结果。</td>
<td>是</td>
</tr>
</tbody>
</table>
<ul>
<li>请求处理成功时，响应体的<code>result</code>具有如下属性：</li>
</ul>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>detectedObjects</code></td>
<td><code>array</code></td>
<td>目标的位置、类别等信息。</td>
</tr>
<tr>
<td><code>image</code></td>
<td><code>string</code></td>
<td>目标检测结果图。图像为JPEG格式，使用Base64编码。</td>
</tr>
</tbody>
</table>
<p><code>detectedObjects</code>中的每个元素为一个<code>object</code>，具有如下属性：</p>
<table>
<thead>
<tr>
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>bbox</code></td>
<td><code>array</code></td>
<td>目标位置。数组中元素依次为边界框左上角x坐标、左上角y坐标、右下角x坐标以及右下角y坐标。</td>
</tr>
<tr>
<td><code>categoryId</code></td>
<td><code>integer</code></td>
<td>目标类别ID。</td>
</tr>
<tr>
<td><code>score</code></td>
<td><code>number</code></td>
<td>目标得分。</td>
</tr>
</tbody>
</table>
<p><code>result</code>示例如下：</p>
<pre><code class="language-json">{
&quot;detectedObjects&quot;: [
{
&quot;bbox&quot;: [
404.4967956542969,
90.15770721435547,
506.2465515136719,
285.4187316894531
],
&quot;categoryId&quot;: 0,
&quot;score&quot;: 0.7418514490127563
},
{
&quot;bbox&quot;: [
155.33145141601562,
81.10954284667969,
199.71136474609375,
167.4235382080078
],
&quot;categoryId&quot;: 1,
&quot;score&quot;: 0.7328268885612488
}
],
&quot;image&quot;: &quot;xxxxxx&quot;
}
</code></pre></details>

<details><summary>多语言调用服务示例</summary>

<details>
<summary>Python</summary>


<pre><code class="language-python">import base64
import requests

API_URL = &quot;http://localhost:8080/small-object-detection&quot; # 服务URL
image_path = &quot;./demo.jpg&quot;
output_image_path = &quot;./out.jpg&quot;

# 对本地图像进行Base64编码
with open(image_path, &quot;rb&quot;) as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode(&quot;ascii&quot;)

payload = {&quot;image&quot;: image_data}  # Base64编码的文件内容或者图像URL

# 调用API
response = requests.post(API_URL, json=payload)

# 处理接口返回数据
assert response.status_code == 200
result = response.json()[&quot;result&quot;]
with open(output_image_path, &quot;wb&quot;) as file:
    file.write(base64.b64decode(result[&quot;image&quot;]))
print(f&quot;Output image saved at {output_image_path}&quot;)
print(&quot;\nDetected objects:&quot;)
print(result[&quot;detectedObjects&quot;])
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &quot;cpp-httplib/httplib.h&quot; // https://github.com/Huiyicc/cpp-httplib
#include &quot;nlohmann/json.hpp&quot; // https://github.com/nlohmann/json
#include &quot;base64.hpp&quot; // https://github.com/tobiaslocker/base64

int main() {
    httplib::Client client(&quot;localhost:8080&quot;);
    const std::string imagePath = &quot;./demo.jpg&quot;;
    const std::string outputImagePath = &quot;./out.jpg&quot;;

    httplib::Headers headers = {
        {&quot;Content-Type&quot;, &quot;application/json&quot;}
    };

    // 对本地图像进行Base64编码
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

    // 调用API
    auto response = client.Post(&quot;/small-object-detection&quot;, headers, body, &quot;application/json&quot;);
    // 处理接口返回数据
    if (response &amp;&amp; response-&gt;status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response-&gt;body);
        auto result = jsonResponse[&quot;result&quot;];

        encodedImage = result[&quot;image&quot;];
        std::string decodedString = base64::from_base64(encodedImage);
        std::vector&lt;unsigned char&gt; decodedImage(decodedString.begin(), decodedString.end());
        std::ofstream outputImage(outPutImagePath, std::ios::binary | std::ios::out);
        if (outputImage.is_open()) {
            outputImage.write(reinterpret_cast&lt;char*&gt;(decodedImage.data()), decodedImage.size());
            outputImage.close();
            std::cout &lt;&lt; &quot;Output image saved at &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        } else {
            std::cerr &lt;&lt; &quot;Unable to open file for writing: &quot; &lt;&lt; outPutImagePath &lt;&lt; std::endl;
        }

        auto detectedObjects = result[&quot;detectedObjects&quot;];
        std::cout &lt;&lt; &quot;\nDetected objects:&quot; &lt;&lt; std::endl;
        for (const auto&amp; category : detectedObjects) {
            std::cout &lt;&lt; category &lt;&lt; std::endl;
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
        String API_URL = &quot;http://localhost:8080/small-object-detection&quot;; // 服务URL
        String imagePath = &quot;./demo.jpg&quot;; // 本地图像
        String outputImagePath = &quot;./out.jpg&quot;; // 输出图像

        // 对本地图像进行Base64编码
        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String imageData = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode params = objectMapper.createObjectNode();
        params.put(&quot;image&quot;, imageData); // Base64编码的文件内容或者图像URL

        // 创建 OkHttpClient 实例
        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.Companion.get(&quot;application/json; charset=utf-8&quot;);
        RequestBody body = RequestBody.Companion.create(params.toString(), JSON);
        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        // 调用API并处理接口返回数据
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode resultNode = objectMapper.readTree(responseBody);
                JsonNode result = resultNode.get(&quot;result&quot;);
                String base64Image = result.get(&quot;image&quot;).asText();
                JsonNode detectedObjects = result.get(&quot;detectedObjects&quot;);

                byte[] imageBytes = Base64.getDecoder().decode(base64Image);
                try (FileOutputStream fos = new FileOutputStream(outputImagePath)) {
                    fos.write(imageBytes);
                }
                System.out.println(&quot;Output image saved at &quot; + outputImagePath);
                System.out.println(&quot;\nDetected objects: &quot; + detectedObjects.toString());
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
    API_URL := &quot;http://localhost:8080/small-object-detection&quot;
    imagePath := &quot;./demo.jpg&quot;
    outputImagePath := &quot;./out.jpg&quot;

    // 对本地图像进行Base64编码
    imageBytes, err := ioutil.ReadFile(imagePath)
    if err != nil {
        fmt.Println(&quot;Error reading image file:&quot;, err)
        return
    }
    imageData := base64.StdEncoding.EncodeToString(imageBytes)

    payload := map[string]string{&quot;image&quot;: imageData} // Base64编码的文件内容或者图像URL
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Println(&quot;Error marshaling payload:&quot;, err)
        return
    }

    // 调用API
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

    // 处理接口返回数据
    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Println(&quot;Error reading response body:&quot;, err)
        return
    }
    type Response struct {
        Result struct {
            Image      string   `json:&quot;image&quot;`
            DetectedObjects []map[string]interface{} `json:&quot;detectedObjects&quot;`
        } `json:&quot;result&quot;`
    }
    var respData Response
    err = json.Unmarshal([]byte(string(body)), &amp;respData)
    if err != nil {
        fmt.Println(&quot;Error unmarshaling response body:&quot;, err)
        return
    }

    outputImageData, err := base64.StdEncoding.DecodeString(respData.Result.Image)
    if err != nil {
        fmt.Println(&quot;Error decoding base64 image data:&quot;, err)
        return
    }
    err = ioutil.WriteFile(outputImagePath, outputImageData, 0644)
    if err != nil {
        fmt.Println(&quot;Error writing image to file:&quot;, err)
        return
    }
    fmt.Printf(&quot;Image saved at %s.jpg\n&quot;, outputImagePath)
    fmt.Println(&quot;\nDetected objects:&quot;)
    for _, category := range respData.Result.DetectedObjects {
        fmt.Println(category)
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
    static readonly string API_URL = &quot;http://localhost:8080/small-object-detection&quot;;
    static readonly string imagePath = &quot;./demo.jpg&quot;;
    static readonly string outputImagePath = &quot;./out.jpg&quot;;

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        // 对本地图像进行Base64编码
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        string image_data = Convert.ToBase64String(imageBytes);

        var payload = new JObject{ { &quot;image&quot;, image_data } }; // Base64编码的文件内容或者图像URL
        var content = new StringContent(payload.ToString(), Encoding.UTF8, &quot;application/json&quot;);

        // 调用API
        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        // 处理接口返回数据
        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        string base64Image = jsonResponse[&quot;result&quot;][&quot;image&quot;].ToString();
        byte[] outputImageBytes = Convert.FromBase64String(base64Image);

        File.WriteAllBytes(outputImagePath, outputImageBytes);
        Console.WriteLine($&quot;Output image saved at {outputImagePath}&quot;);
        Console.WriteLine(&quot;\nDetected objects:&quot;);
        Console.WriteLine(jsonResponse[&quot;result&quot;][&quot;detectedObjects&quot;].ToString());
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');

const API_URL = 'http://localhost:8080/small-object-detection'
const imagePath = './demo.jpg'
const outputImagePath = &quot;./out.jpg&quot;;

let config = {
   method: 'POST',
   maxBodyLength: Infinity,
   url: API_URL,
   data: JSON.stringify({
    'image': encodeImageToBase64(imagePath)  // Base64编码的文件内容或者图像URL
  })
};

// 对本地图像进行Base64编码
function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

// 调用API
axios.request(config)
.then((response) =&gt; {
    // 处理接口返回数据
    const result = response.data[&quot;result&quot;];
    const imageBuffer = Buffer.from(result[&quot;image&quot;], 'base64');
    fs.writeFile(outputImagePath, imageBuffer, (err) =&gt; {
      if (err) throw err;
      console.log(`Output image saved at ${outputImagePath}`);
    });
    console.log(&quot;\nDetected objects:&quot;);
    console.log(result[&quot;detectedObjects&quot;]);
})
.catch((error) =&gt; {
  console.log(error);
});
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = &quot;http://localhost:8080/small-object-detection&quot;; // 服务URL
$image_path = &quot;./demo.jpg&quot;;
$output_image_path = &quot;./out.jpg&quot;;

// 对本地图像进行Base64编码
$image_data = base64_encode(file_get_contents($image_path));
$payload = array(&quot;image&quot; =&gt; $image_data); // Base64编码的文件内容或者图像URL

// 调用API
$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

// 处理接口返回数据
$result = json_decode($response, true)[&quot;result&quot;];
file_put_contents($output_image_path, base64_decode($result[&quot;image&quot;]));
echo &quot;Output image saved at &quot; . $output_image_path . &quot;\n&quot;;
echo &quot;\nDetected objects:\n&quot;;
print_r($result[&quot;detectedObjects&quot;]);

?&gt;
</code></pre></details>
</details>
<br/>

📱 <b>端侧部署</b>：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../../../pipeline_deploy/edge_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。


## 4. 二次开发
如果通用小目标检测产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用<b>您自己拥有的特定领域或应用场景的数据</b>对现有模型进行进一步的<b>微调</b>，以提升小目标检测产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用小目标检测产线包含小目标检测模块，如果模型产线的效果不及预期，那么您需要参考[小目标检测模块开发教程](../../../module_usage/tutorials/cv_modules/small_object_detection.md)中的[二次开发](../../../module_usage/tutorials/cv_modules/small_object_detection.md#四二次开发)章节，使用您的私有数据集对小目标检测模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```python
......
Pipeline:
  model: PP-YOLOE_plus_SOD-L  #可修改为微调后模型的本地路径
  batch_size: 1
  device: "gpu:0"
......
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，<b>仅需修改 `--device` 参数</b>即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行小目标检测产线的推理，使用的 Python 命令为：

```bash
paddlex --pipeline multilabel_classification --input small_object_detection.jpg --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 `--device` 修改为 npu:0 即可：

```bash
paddlex --pipeline multilabel_classification --input small_object_detection.jpg --device npu:0
```
若您想在更多种类的硬件上使用通用小目标检测产线，请参考[PaddleX多硬件使用指南](../../../other_devices_support/multi_devices_use_guide.md)。
