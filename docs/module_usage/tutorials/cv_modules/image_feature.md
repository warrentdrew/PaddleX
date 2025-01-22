---
comments: true
---

# 图像特征模块使用教程

## 一、概述
图像特征模块是计算机视觉中的一项重要任务之一，主要指的是通过深度学习方法自动从图像数据中提取有用的特征，以便于后续的图像检索任务。该模块的性能直接影响到后续任务的准确性和效率。在实际应用中，图像特征通常会输出一组特征向量，这些向量能够有效地表示图像的内容、结构、纹理等信息，并将作为输入传递给后续的检索模块进行处理。

## 二、支持模型列表


<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>recall@1 (%)</th>
<th>GPU推理耗时 (ms)</th>
<th>CPU推理耗时 (ms)</th>
<th>模型存储大小 (M)</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-ShiTuV2_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-ShiTuV2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_pretrained.pdparams">训练模型</a></td>
<td>84.2</td>
<td>5.23428</td>
<td>19.6005</td>
<td>16.3 M</td>
<td rowspan="3">PP-ShiTuV2是一个通用图像特征系统，由主体检测、特征提取、向量检索三个模块构成，这些模型是其中的特征提取模块的模型之一，可以根据系统的情况选择不同的模型。</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_base</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-ShiTuV2_rec_CLIP_vit_base_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_base_pretrained.pdparams">训练模型</a></td>
<td>88.69</td>
<td>13.1957</td>
<td>285.493</td>
<td>306.6 M</td>
</tr>
<tr>
<td>PP-ShiTuV2_rec_CLIP_vit_large</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-ShiTuV2_rec_CLIP_vit_large_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-ShiTuV2_rec_CLIP_vit_large_pretrained.pdparams">训练模型</a></td>
<td>91.03</td>
<td>51.1284</td>
<td>1131.28</td>
<td>1.05 G</td>
</tr>
</table>


<b>注：以上精度指标为 AliProducts recall@1。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。</b>

## 三、快速集成
> ❗ 在快速集成前，请先安装 PaddleX 的 wheel 包，详细请参考 [PaddleX本地安装教程](../../../installation/installation.md)

完成 wheel 包的安装后，几行代码即可完成图像特征模块的推理，可以任意切换该模块下的模型，您也可以将图像特征的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_recognition_001.jpg)到本地。

```python
from paddlex import create_model

model_name = "PP-ShiTuV2_rec"
model = create_model(model_name)
output = model.predict("general_image_recognition_001.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_json("./output/res.json")
```
<details><summary>👉 <b>运行后，得到的结果为：（点击展开）</b></summary>

```bash
{'res': {'input_path': 'general_image_recognition_001.jpg', 'feature': [0.049109019339084625, 0.003743218956515193, 0.005219039972871542, -0.02721826359629631, -0.016909820958971977, -0.006206876132637262, -0.04125503450632095, -0.0019930789712816477, -0.022831404581665993, -0.047313764691352844, 0.060403138399124146, 0.0761566013097763, 0.0017959520919248462, 0.0493767075240612, -0.021170074120163918, -0.03655105456709862, 0.02802395075559616, 0.039860036224126816, -0.02787216380238533, 0.02957169897854328, 0.06538619846105576, 0.006429907865822315, 0.007104719523340464, -0.005089965183287859, -0.05652903392910957, -0.005592004396021366, 0.03234156593680382, 0.07858140766620636, 0.031976014375686646, 0.009858119301497936, -0.005774488672614098, -0.06363064795732498, 0.022391995415091515, 0.01687457598745823, -0.023193085566163063, -0.04410340636968613, -0.048582229763269424, -0.005215164739638567, -0.045220136642456055, 0.02273370884358883, 0.1008470356464386, 0.020996153354644775, 0.04300517961382866, 0.012766947969794273, 0.0331314392387867, -0.025111157447099686, 0.04866917431354523, -0.0007637099479325116, 0.09444739669561386, -0.11289195716381073, -0.07322289794683456, -0.0777730941772461, -0.015321447513997555, -0.004911054391413927, 0.007349034305661917, -0.06889548897743225, -0.011488647200167179, 0.0266891997307539, 0.015688180923461914, -0.06728753447532654, 0.03612099587917328, 0.025985529646277428, 0.03768249973654747, -0.08363037556409836, -0.048576220870018005, 0.08420056849718094, 0.023980043828487396, 0.07883623242378235, -0.013208202086389065, -0.018564118072390556, -0.017457934096455574, -0.020058095455169678, -0.02541436441242695, -0.008189533837139606, -0.056144408881664276, -0.02954513393342495, 0.010910517536103725, 0.003301263554021716, 0.06621509790420532, -0.06254079937934875, 0.06691820919513702, -0.061315152794122696, -0.0664873942732811, 0.03236202895641327, 0.034841813147068024, -0.013619803823530674, 0.10573607683181763, 0.02072797901928425, 0.055478695780038834, 0.010184620507061481, -0.05034280940890312, -0.02191540040075779, 0.047144751995801926, -0.031101807951927185, -0.016248639672994614, 0.05873332917690277, 0.015815891325473785, -0.001074428902938962, -0.010065405629575253, -0.045008182525634766, 0.03385455906391144, 0.0015282221138477325, -0.033003728836774826, -0.03174556791782379, 0.11101125925779343, -0.006336087826639414, 0.01115291379392147, 0.05289424583315849, -0.031148016452789307, -0.025501884520053864, -0.04285573586821556, -0.025347966700792313, 0.07360142469406128, -0.021775048226118088, 0.06776975840330124, -0.027090832591056824, 0.004754350520670414, 0.020962830632925034, -0.015184056013822556, 0.0010469526750966907, 0.10906203091144562, 0.023142946884036064, -0.014811648055911064, 0.060142189264297485, 0.00820994470268488, -0.02131684496998787, 0.0638369619846344, -0.04250387102365494, 0.008871919475495815, -0.008007975295186043, 0.07150974124670029, 0.011682109907269478, -0.006690092850476503, 0.011732078157365322, -0.07107444852590561, -0.06642609089612961, -0.011343806982040405, -0.006821080576628447, -0.02998213842511177, 0.028652319684624672, 0.03125355765223503, 0.07194835692644119, 0.07021243870258331, 0.002937359269708395, -0.0499102883040905, 0.040573444217443466, -0.019794275984168053, 0.03276952728629112, -0.11937176436185837, -0.05637907609343529, -0.016590707004070282, -0.020010240375995636, 0.026032714173197746, 0.0035290969535708427, -0.02005072496831417, 0.010683903470635414, 0.016347797587513924, 0.015019580721855164, -0.013746236450970173, 0.059569939970970154, -0.049140941351652145, -0.0019614780321717262, -0.08775275200605392, -0.10722333192825317, -0.042984429746866226, -0.00020981401030439883, 0.019572105258703232, -0.06898674368858337, 0.01378229632973671, 0.010090887546539307, 0.004062692169100046, 0.03605732321739197, -0.030028242617845535, -0.071064792573452, 2.3956217773957178e-05, -0.04649794474244118, 0.006212098523974419, 0.05053247511386871, 0.017688272520899773, 0.06759831309318542, 0.06286999583244324, -0.0010359658626839519, 0.02886095643043518, 0.07879934459924698, -0.04438954219222069, 0.03000231273472309, 0.0032854368910193443, 0.04237217828631401, -0.019776295870542526, -0.003197435289621353, -0.029772961512207985, 0.014659246429800987, 0.007493179757148027, -0.05654020234942436, -0.06438001990318298, 0.09076389670372009, 0.06214659661054611, 0.004840471316128969, 0.045114848762750626, 0.07397229224443436, -0.032566577196121216, -0.02464713528752327, -0.001989303156733513, 0.011997431516647339, -0.05213696509599686, -0.016684064641594887, -0.025001073256134987, -0.06234518438577652, 0.08302164077758789, 0.06388438493013382, -0.02603762038052082, -0.057507626712322235, 0.010737594217061996, 0.021288368850946426, 0.050199754536151886, 0.020688340067863464, -0.03297201544046402, 0.046142708510160446, -0.010062780231237411, -0.009058497846126556, -0.028288882225751877, 0.04905378818511963, 0.014915363863110542, 0.013268127106130123, 0.0682050809264183, -0.05951741710305214, 0.057072725147008896, -0.05045686662197113, -0.06781881302595139, 0.013548677787184715, 0.05480438843369484, 0.004949226509779692, -0.06020767241716385, 0.06817059963941574, -0.03284472972154617, 0.014837299473583698, -0.02967672049999237, 0.01816580630838871, 0.02033187262713909, 0.02823079377412796, -0.04500351473689079, 0.08674795180559158, -0.029325610026717186, 0.023551611229777336, 0.06388993561267853, -0.009744416922330856, -0.03457322716712952, 0.04782603681087494, 0.02554101124405861, -0.0456324964761734, -0.028274627402424812, 0.06662072241306305, -0.05609177425503731, 0.05615284666419029, 0.019022393971681595, -0.0025437776930630207, 0.009997396729886532, -0.02311607636511326, -0.01859533227980137, 0.014079087413847446, -0.004390522837638855, 0.033753011375665665, -0.01057991199195385, 0.06405404955148697, -0.044001828879117966, 0.030532050877809525, 0.07704707980155945, -0.008154668845236301, 0.07150844484567642, -0.025875508785247803, 0.012938959524035454, 0.031110644340515137, 0.040104109793901443, -0.015837285667657852, 0.013013459742069244, -0.04932061582803726, 0.1163453534245491, -0.018216004595160484, 0.09031085669994354, -0.00798568595200777, 0.03739010915160179, 0.009020711295306683, 0.0121764512732625, 0.05221538990736008, -0.039455823600292206, 0.0006922244210727513, 0.015570797957479954, 0.07312478870153427, -0.020688941702246666, -0.013934718444943428, 0.008533086627721786, 0.06440478563308716, -0.01079096831381321, 0.01836889237165451, 0.02253551408648491, 0.013003944419324398, -0.09377135336399078, -0.0679609626531601, 0.08662433922290802, 0.04313892871141434, -0.0991244912147522, 0.031046167016029358, -0.00529451621696353, 0.03583913296461105, -0.036976899951696396, -0.02047165483236313, 0.05907558649778366, -0.01609981805086136, 0.025996552780270576, -0.020804448053240776, -0.08788128197193146, 0.008795336820185184, 0.02871711179614067, 0.03529304638504982, 0.03294558450579643, -0.042029064148664474, 0.06038316339254379, -0.04894087836146355, 0.062449462711811066, 0.006345283705741167, -0.05110999941825867, -0.05251497030258179, -0.0485706627368927, 0.01973171904683113, 0.008282365277409554, -0.07566660642623901, -0.054833680391311646, -0.07128996402025223, 0.04532742500305176, 0.03617053106427193, -0.08390213549137115, 0.059522684663534164, -0.0420474074780941, -0.01778397522866726, -0.0009484086185693741, -0.0005348647246137261, 0.060148268938064575, -0.0330856516957283, -0.0246503297239542, 0.017774641513824463, 0.030457578599452972, 0.05355843901634216, -0.0587775893509388, -0.10057137906551361, 0.018445275723934174, -0.009815521538257599, -0.018039006739854813, 0.07852229475975037, 0.06736239790916443, -0.001333046704530716, 0.007442455273121595, 0.024762822315096855, 0.01432487741112709, -0.015761420130729675, -0.06996625661849976, -0.046995896846055984, 0.007052265573292971, 0.01292750146239996, -0.040947142988443375, -0.04076245054602623, 0.02297302521765232, 0.031471725553274155, -0.00020813643641304225, 0.038925208151340485, -0.06558586657047272, -0.007383601740002632, -0.023945683613419533, -0.00975192990154028, -0.02227136865258217, 0.04719878360629082, -0.06225449591875076, -0.01973516121506691, -0.03993571922183037, 0.05859784781932831, -0.033084310591220856, -0.09690506756305695, -0.023898473009467125, -0.021453123539686203, -0.03663061931729317, -0.017285870388150215, 0.03141544386744499, 0.03807565197348595, -0.026278872042894363, -0.01809430494904518, -0.015841489657759666, 0.012133757583796978, 0.015615759417414665, 0.025031639263033867, 0.024692615494132042, 0.04616452008485794, -0.060463856905698776, -0.015853295102715492, -0.018289266154170036, -0.04386031627655029, 0.03577912971377373, -0.027260515838861465, 0.0262057613581419, 0.057801421731710434, -0.03247777372598648, -0.025150176137685776, 0.00601597735658288, 0.04657217860221863, -0.080239437520504, 0.03466523066163063, -0.030453147366642952, 0.0482921376824379, -0.08164504170417786, -0.014207101427018642, 0.009327770210802555, -0.04090266674757004, 0.030108662322163582, 0.009122633375227451, -0.014895373024046421, -0.05183123052120209, -0.013794025406241417, 0.07302702963352203, -0.09872224181890488, -0.018709152936935425, 0.0033975779078900814, -0.014089024625718594, 0.03661659359931946, 0.057056576013565063, 0.020562296733260155, 0.043459419161081314, 0.01512663159519434, -0.08066604286432266, -0.004175281152129173, -0.01924264058470726, 0.019628198817372322, 0.0038949784357100725, 0.0417584627866745, 0.04380005970597267, 0.07349026203155518, 0.014636299572885036, -0.05216812714934349, -0.012727170251309872, 0.045851919800043106, -0.01108911819756031, 0.03997885435819626, -0.030407968908548355, -0.010511808097362518, -0.05589114874601364, 0.044310227036476135, 0.07810009270906448, -0.04502151906490326, -0.04358488321304321, -0.05491747707128525, -0.03352531045675278, 0.05367979034781456, -0.051684021949768066, -0.09159758687019348, -0.06795314699411392, 0.05151400715112686, -0.03667590394616127, 0.0415973998606205, 0.004540375899523497, 0.05324205011129379, 0.007202384993433952, -0.04345241189002991, 0.050976917147636414, -0.04033355042338371, 0.06330423802137375, 0.05229709669947624, 0.07360764592885971, -0.032564569264650345, 0.05199885368347168, -0.00532008009031415, -0.0022711465135216713, 0.02948114089667797, -0.016920195892453194, -0.02260630950331688, 0.028158463537693024, 0.07792772352695465, -0.018052909523248672, -0.03556442633271217, 0.00789704080671072, -0.07626558840274811, -0.03160900995135307, 0.01614975929260254, 0.029233204200863838, -0.019288279116153717, 0.042064376175403595, 0.02296893298625946, -0.02441602759063244, 0.0140019990503788, -0.06483341753482819, 0.11968966573476791, 0.03714313730597496, -0.06824050098657608, -0.016793427988886833, 0.009564070962369442, -0.09704745560884476, 0.01676984317600727, -0.02669912949204445, 0.0003774994402192533, -0.06364470720291138, 0.01929471455514431, -0.009203671477735043, -0.003950135316699743, -0.03651873394846916, -0.013927181251347065, -0.0008955051307566464, 0.02604137919843197, -0.07753892242908478, 0.015690961852669716, 0.0001985478593269363, -0.029516037553548813, -0.028976714238524437, -0.07917939871549606, 0.036289919167757034, 0.03013240359723568, -0.07124550640583038]}}
```

参数含义如下：
- `input_path`：输入的待预测图像的路径
- `feature`：表示特征向量的浮点数列表，长度为512

</details>

相关方法、参数等说明如下：

* `create_model`实例化图像特征模型（此处以`PP-ShiTuV2_rec`为例），具体说明如下：
<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>model_name</code></td>
<td>模型名称</td>
<td><code>str</code></td>
<td>无</td>
<td>无</td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径</td>
<td><code>str</code></td>
<td>无</td>
<td>无</td>
</tr>
</table>

* 其中，`model_name` 必须指定，指定 `model_name` 后，默认使用 PaddleX 内置的模型参数，在此基础上，指定 `model_dir` 时，使用用户自定义的模型。

* 调用图像特征模型的 `predict()` 方法进行推理预测，`predict()` 方法参数有 `input`和`batch_size`具体说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>可选项</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型</td>
<td><code>Python Var</code>/<code>str</code>/<code>dict</code>/<code>list</code></td>
<td>
<ul>
  <li><b>Python变量</b>，如<code>numpy.ndarray</code>表示的图像数据</li>
  <li><b>文件路径</b>，如图像文件的本地路径：<code>/root/data/img.jpg</code></li>
  <li><b>URL链接</b>，如图像文件的网络URL：<a href = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">示例</a></li>
  <li><b>本地目录</b>，该目录下需包含待预测数据文件，如本地路径：<code>/root/data/</code></li>
  <li><b>字典</b>，字典的<code>key</code>需与具体任务对应，如图像分类任务对应<code>\"img\"</code>，字典的<code>val</code>支持上述类型数据，例如：<code>{\"img\": \"/root/data1\"}</code></li>
  <li><b>列表</b>，列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>，<code>[\"/root/data1\", \"/root/data2\"]</code>，<code>[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]</code></li>
</ul>
</td>
<td>无</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>批大小</td>
<td><code>int</code></td>
<td>任意整数</td>
<td>1</td>
</tr>
</table>

* 对预测结果进行处理，每个样本的预测结果均为`dict`类型，且支持打印、保存为图片、保存为`json`文件的操作:

<table>
<thead>
<tr>
<th>方法</th>
<th>方法说明</th>
<th>参数</th>
<th>参数类型</th>
<th>参数说明</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td rowspan = "3"><code>print()</code></td>
<td rowspan = "3">打印结果到终端</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan = "3"><code>save_to_json()</code></td>
<td rowspan = "3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
</table>

* 此外，也支持通过属性获取预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan = "1"><code>json</code></td>
<td rowspan = "1">获取预测的<code>json</code>格式的结果</td>
</tr>
</table>

## 四、二次开发
如果你追求更高精度的现有模型，可以使用 PaddleX 的二次开发能力，开发更好的图像特征模型。在使用 PaddleX 开发图像特征模型之前，请务必安装 PaddleX的分类相关模型训练插件，安装过程可以参考 [PaddleX本地安装教程](../../../installation/installation.md)

### 4.1 数据准备
在进行模型训练前，需要准备相应任务模块的数据集。PaddleX 针对每一个模块提供了数据校验功能，<b>只有通过数据校验的数据才可以进行模型训练</b>。此外，PaddleX 为每一个模块都提供了 Demo 数据集，您可以基于官方提供的 Demo 数据完成后续的开发。若您希望用私有数据集进行后续的模型训练，可以参考[PaddleX图像特征任务模块数据标注教程](../../../data_annotations/cv_modules/image_feature.md)

#### 4.1.1 Demo 数据下载
您可以参考下面的命令将 Demo 数据集下载到指定文件夹：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Inshop_examples.tar -P ./dataset
tar -xf ./dataset/Inshop_examples.tar -C ./dataset/
```
#### 4.1.2 数据校验
一行命令即可完成数据校验：

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
执行上述命令后，PaddleX 会对数据集进行校验，并统计数据集的基本信息，命令运行成功后会在log中打印出`Check dataset passed !`信息。校验结果文件保存在`./output/check_dataset_result.json`，同时相关产出会保存在当前目录的`./output/check_dataset`目录下，产出目录中包括可视化的示例样本图片和样本分布直方图。

<details><summary>👉 <b>校验结果详情（点击展开）</b></summary>

<p>校验结果文件具体内容为：</p>
<pre><code class="language-bash">
  &quot;done_flag&quot;: true,
  &quot;check_pass&quot;: true,
  &quot;attributes&quot;: {
    &quot;train_samples&quot;: 1000,
    &quot;train_sample_paths&quot;: [
      &quot;check_dataset/demo_img/05_1_front.jpg&quot;,
      &quot;check_dataset/demo_img/02_1_front.jpg&quot;,
      &quot;check_dataset/demo_img/02_3_back.jpg&quot;,
      &quot;check_dataset/demo_img/04_3_back.jpg&quot;,
      &quot;check_dataset/demo_img/04_2_side.jpg&quot;,
      &quot;check_dataset/demo_img/12_1_front.jpg&quot;,
      &quot;check_dataset/demo_img/07_2_side.jpg&quot;,
      &quot;check_dataset/demo_img/04_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/04_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/01_1_front.jpg&quot;
    ],
    &quot;gallery_samples&quot;: 110,
    &quot;gallery_sample_paths&quot;: [
      &quot;check_dataset/demo_img/06_2_side.jpg&quot;,
      &quot;check_dataset/demo_img/01_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/04_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/02_1_front.jpg&quot;,
      &quot;check_dataset/demo_img/02_3_back.jpg&quot;,
      &quot;check_dataset/demo_img/02_3_back.jpg&quot;,
      &quot;check_dataset/demo_img/02_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/03_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/02_2_side.jpg&quot;,
      &quot;check_dataset/demo_img/03_2_side.jpg&quot;
    ],
    &quot;query_samples&quot;: 125,
    &quot;query_sample_paths&quot;: [
      &quot;check_dataset/demo_img/08_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/01_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/02_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/04_4_full.jpg&quot;,
      &quot;check_dataset/demo_img/09_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/04_3_back.jpg&quot;,
      &quot;check_dataset/demo_img/02_1_front.jpg&quot;,
      &quot;check_dataset/demo_img/06_2_side.jpg&quot;,
      &quot;check_dataset/demo_img/02_7_additional.jpg&quot;,
      &quot;check_dataset/demo_img/02_2_side.jpg&quot;
    ]
  },
  &quot;analysis&quot;: {
    &quot;histogram&quot;: &quot;check_dataset/histogram.png&quot;
  },
  &quot;dataset_path&quot;: &quot;./dataset/Inshop_examples&quot;,
  &quot;show_type&quot;: &quot;image&quot;,
  &quot;dataset_type&quot;: &quot;ShiTuRecDataset&quot;
}
</code></pre>
<p>上述校验结果中，check_pass  为 true 表示数据集格式符合要求，其他部分指标的说明如下：</p>
<ul>
<li><code>attributes.train_samples</code>：该数据集训练样本数量为 1000；</li>
<li><code>attributes.gallery_samples</code>：该数据集被查询样本数量为 110；</li>
<li><code>attributes.query_samples</code>：该数据集查询样本数量为 125；</li>
<li><code>attributes.train_sample_paths</code>：该数据集训练样本可视化图片相对路径列表；</li>
<li><code>attributes.gallery_sample_paths</code>：该数据集被查询样本可视化图片相对路径列表；</li>
<li><code>attributes.query_sample_paths</code>：该数据集查询样本可视化图片相对路径列表；
另外，数据集校验还对数据集中图像数量和图像类别情况进行了分析，并绘制了分布直方图（histogram.png）：</li>
</ul>
<p><img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/modules/img_recognition/01.png"></p></details>

### 4.1.3 数据集格式转换/数据集划分（可选）
在您完成数据校验之后，可以通过<b>修改配置文件</b>或是<b>追加超参数</b>的方式对数据集的格式进行转换，也可以对数据集的训练/验证比例进行重新划分。

<details><summary>👉 <b>格式转换/数据集划分详情（点击展开）</b></summary>

<p><b>（1）数据集格式转换</b></p>
<p>图像特征任务支持 <code>LabelMe</code>格式的数据集转换为 <code>ShiTuRecDataset</code>格式，数据集格式转换的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>convert</code>:</li>
<li><code>enable</code>: 是否进行数据集格式转换，图像特征任务支持 <code>LabelMe</code>格式的数据集转换为 <code>ShiTuRecDataset</code>格式，默认为 <code>False</code>;</li>
<li><code>src_dataset_type</code>: 如果进行数据集格式转换，则需设置源数据集格式，默认为 <code>null</code>，可选值为 <code>LabelMe</code> ；
例如，您想将<code>LabelMe</code>格式的数据集转换为 <code>ShiTuRecDataset</code>格式，则需将配置文件修改为：</li>
</ul>
<pre><code class="language-bash">cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/image_classification_labelme_examples.tar -P ./dataset
tar -xf ./dataset/image_classification_labelme_examples.tar -C ./dataset/
</code></pre>
<pre><code class="language-bash">......
CheckDataset:
  ......
  convert:
    enable: True
    src_dataset_type: LabelMe
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c  paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/image_classification_labelme_examples
</code></pre>
<p>数据转换执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/image_classification_labelme_examples \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
</code></pre>
<p><b>（2）数据集划分</b></p>
<p>数据集划分的参数可以通过修改配置文件中 <code>CheckDataset</code> 下的字段进行设置，配置文件中部分参数的示例说明如下：</p>
<ul>
<li><code>CheckDataset</code>:</li>
<li><code>split</code>:</li>
<li><code>enable</code>: 是否进行重新划分数据集，为 <code>True</code> 时进行数据集格式转换，默认为 <code>False</code>；</li>
<li><code>train_percent</code>: 如果重新划分数据集，则需要设置训练集的百分比，类型为 0-100 之间的任意整数，需要保证和 <code>gallery_percent 、query_percent</code> 值加和为100；</li>
</ul>
<p>例如，您想重新划分数据集为 训练集占比70%、被查询数据集占比20%，查询数据集占比10%，则需将配置文件修改为：</p>
<pre><code class="language-bash">......
CheckDataset:
  ......
  split:
    enable: True
    train_percent: 70
    gallery_percent: 20
    query_percent: 10
  ......
</code></pre>
<p>随后执行命令：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples
</code></pre>
<p>数据划分执行之后，原有标注文件会被在原路径下重命名为 <code>xxx.bak</code>。</p>
<p>以上参数同样支持通过追加命令行参数的方式进行设置：</p>
<pre><code class="language-bash">python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/Inshop_examples \
    -o CheckDataset.split.enable=True \
    -o CheckDataset.split.train_percent=70 \
    -o CheckDataset.split.gallery_percent=20 \
    -o CheckDataset.split.query_percent=10
</code></pre>
<blockquote>
<p>❗注意 ：由于图像特征模型评估的特殊性，当且仅当 train、query、gallery 集合属于同一类别体系下，数据切分才有意义，在图像特征模的评估过程中，必须满足 gallery 集合和 query 集合属于同一类别体系，其允许和 train 集合不在同一类别体系， 如果 gallery 集合和 query 集合与 train 集合不在同一类别体系，则数据划分后的评估没有意义，建议谨慎操作。</p>
</blockquote></details>

### 4.2 模型训练
一条命令即可完成模型的训练，以此处图像特征模型 PP-ShiTuV2_rec 的训练为例：

```
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-ShiTuV2_rec.yaml`，训练其他模型时，需要的指定相应的配置文件，模型和配置的文件的对应关系，可以查阅[PaddleX模型列表（CPU/GPU）](../../../support_list/models_list.md)）
* 指定模式为模型训练：`-o Global.mode=train`
* 指定训练数据集路径：`-o Global.dataset_dir`
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Train`下的字段来进行设置，也可以通过在命令行中追加参数来进行调整。如指定前 2 卡 gpu 训练：`-o Global.device=gpu:0,1`；设置训练轮次数为 10：`-o Train.epochs_iters=10`。更多可修改的参数及其详细解释，可以查阅模型对应任务模块的配置文件说明[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<ul>
<li>模型训练过程中，PaddleX 会自动保存模型权重文件，默认为<code>output</code>，如需指定保存路径，可通过配置文件中 <code>-o Global.output</code> 字段进行设置。</li>
<li>PaddleX 对您屏蔽了动态图权重和静态图权重的概念。在模型训练的过程中，会同时产出动态图和静态图的权重，在模型推理时，默认选择静态图权重推理。</li>
<li>
<p>在完成模型训练后，所有产出保存在指定的输出目录（默认为<code>./output/</code>）下，通常有以下产出：</p>
</li>
<li>
<p><code>train_result.json</code>：训练结果记录文件，记录了训练任务是否正常完成，以及产出的权重指标、相关文件路径等；</p>
</li>
<li><code>train.log</code>：训练日志文件，记录了训练过程中的模型指标变化、loss 变化等；</li>
<li><code>config.yaml</code>：训练配置文件，记录了本次训练的超参数的配置；</li>
<li><code>.pdparams</code>、<code>.pdema</code>、<code>.pdopt.pdstate</code>、<code>.pdiparams</code>、<code>.pdmodel</code>：模型权重相关文件，包括网络参数、优化器、EMA、静态图网络参数、静态图网络结构等；</li>
</ul></details>

## <b>4.3 模型评估</b>
在完成模型训练后，可以对指定的模型权重文件在验证集上进行评估，验证模型精度。使用 PaddleX 进行模型评估，一条命令即可完成模型的评估：

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/Inshop_examples
```
与模型训练类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-ShiTuV2_rec.yaml`）
* 指定模式为模型评估：`-o Global.mode=evaluate`
* 指定验证数据集路径：`-o Global.dataset_dir`.
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Evaluate`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

<details><summary>👉 <b>更多说明（点击展开）</b></summary>

<p>在模型评估时，需要指定模型权重文件路径，每个配置文件中都内置了默认的权重保存路径，如需要改变，只需要通过追加命令行参数的形式进行设置即可，如<code>-o Evaluate.weight_path=./output/best_model/best_model.pdparams</code>。</p>
<p>在完成模型评估后，会产出<code>evaluate_result.json，其记录了</code>评估的结果，具体来说，记录了评估任务是否正常完成，以及模型的评估指标，包含 recall1、recall5、mAP；</p></details>

### <b>4.4 模型推理</b>
在完成模型的训练和评估后，即可使用训练好的模型权重进行推理预测或者进行 Python 集成。

#### 4.4.1 模型推理
通过命令行的方式进行推理预测，只需如下一条命令。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_recognition_001.jpg)到本地。

```bash
python main.py -c paddlex/configs/modules/image_feature/PP-ShiTuV2_rec.yaml  \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/best_model/inference" \
    -o Predict.input="general_image_recognition_001.jpg"
```
与模型训练和评估类似，需要如下几步：

* 指定模型的`.yaml` 配置文件路径（此处为`PP-ShiTuV2_rec.yaml`）
* 指定模式为模型推理预测：`-o Global.mode=predict`
* 指定模型权重路径：`-o Predict.model_dir="./output/best_model/inference"`
* 指定输入数据路径：`-o Predict.input="..."`.
其他相关参数均可通过修改`.yaml`配置文件中的`Global`和`Predict`下的字段来进行设置，详细请参考[PaddleX通用模型配置文件参数说明](../../instructions/config_parameters_common.md)。

> ❗ 注意：图像特征模型的推理结果为一组向量，需要配合检索模块完成图像的识别。

#### 4.4.2 模型集成
模型可以直接集成到 PaddleX 产线中，也可以直接集成到您自己的项目中。

1.<b>产线集成</b>

图像特征模块可以集成的 PaddleX 产线有<b>通用图像特征产线</b>(comming soon)，只需要替换模型路径即可完成相关产线的图像特征模块的模型更新。在产线集成中，你可以使用服务化部署来部署你得到的模型。

2.<b>模块集成</b>

您产出的权重可以直接集成到图像特征模块中，可以参考[快速集成](#三快速集成)的 Python 示例代码，只需要将模型替换为你训练的到的模型路径即可。
