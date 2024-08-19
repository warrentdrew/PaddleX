# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
from typing import Union
from paddleclas.ppcls.utils.config import get_config, override_config

from ...base import BaseConfig
from ....utils.misc import abspath


class ClsConfig(BaseConfig):
    """Image Classification Task Config"""

    def update(self, list_like_obj: list):
        """update self

        Args:
            list_like_obj (list): list of pairs(key0.key1.idx.key2=value), such as:
                [
                    'topk=2',
                    'VALID.transforms.1.ResizeImage.resize_short=300'
                ]
        """
        dict_ = override_config(self.dict, list_like_obj)
        self.reset_from_dict(dict_)

    def load(self, config_file_path: str):
        """load config from yaml file

        Args:
            config_file_path (str): the path of yaml file.

        Raises:
            TypeError: the content of yaml file `config_file_path` error.
        """
        dict_ = yaml.load(open(config_file_path, 'rb'), Loader=yaml.Loader)
        if not isinstance(dict_, dict):
            raise TypeError
        self.reset_from_dict(dict_)

    def dump(self, config_file_path: str):
        """dump self to yaml file

        Args:
            config_file_path (str): the path to save self as yaml file.
        """
        with open(config_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict, f, default_flow_style=False, sort_keys=False)

    def update_dataset(
            self,
            dataset_path: str,
            dataset_type: str=None,
            *,
            train_list_path: str=None, ):
        """update dataset settings

        Args:
            dataset_path (str): the root path of dataset.
            dataset_type (str, optional): dataset type. Defaults to None.
            train_list_path (str, optional): the path of train dataset annotation file . Defaults to None.

        Raises:
            ValueError: the dataset_type error.
        """
        dataset_path = abspath(dataset_path)
        if dataset_type is None:
            dataset_type = 'ClsDataset'
        if train_list_path:
            train_list_path = f"{train_list_path}"
        else:
            train_list_path = f"{dataset_path}/train.txt"

        if dataset_type in ['ClsDataset']:
            ds_cfg = [
                f'DataLoader.Train.dataset.name={dataset_type}',
                f'DataLoader.Train.dataset.image_root={dataset_path}',
                f'DataLoader.Train.dataset.cls_label_path={train_list_path}',
                f'DataLoader.Eval.dataset.name={dataset_type}',
                f'DataLoader.Eval.dataset.image_root={dataset_path}',
                f'DataLoader.Eval.dataset.cls_label_path={dataset_path}/val.txt',
                f'Infer.PostProcess.class_id_map_file={dataset_path}/label.txt'
            ]
        else:
            raise ValueError(f"{repr(dataset_type)} is not supported.")
        self.update(ds_cfg)

    def update_batch_size(self, batch_size: int, mode: str='train'):
        """update batch size setting

        Args:
            batch_size (int): the batch size number to set.
            mode (str, optional): the mode that to be set batch size, must be one of 'train', 'eval', 'test'.
                Defaults to 'train'.

        Raises:
            ValueError: `mode` error.
        """
        if mode == 'train':
            if self.DataLoader["Train"]["sampler"].get("batch_size", False):
                _cfg = [f'DataLoader.Train.sampler.batch_size={batch_size}']
            else:
                _cfg = [f'DataLoader.Train.sampler.first_bs={batch_size}']
                _cfg = [f'DataLoader.Train.dataset.name=MultiScaleDataset']
        elif mode == 'eval':
            _cfg = [f'DataLoader.Eval.sampler.batch_size={batch_size}']
        elif mode == 'test':
            _cfg = [f'DataLoader.Infer.batch_size={batch_size}']
        else:
            raise ValueError("The input `mode` should be train, eval or test.")
        self.update(_cfg)

    def update_learning_rate(self, learning_rate: float):
        """update learning rate

        Args:
            learning_rate (float): the learning rate value to set.
        """
        _cfg = [f'Optimizer.lr.learning_rate={learning_rate}']
        self.update(_cfg)

    def update_warmup_epochs(self, warmup_epochs: int):
        """update warmup epochs

        Args:
            warmup_epochs (int): the warmup epochs value to set.
        """
        _cfg = [f'Optimizer.lr.warmup_epoch={warmup_epochs}']
        self.update(_cfg)

    def update_pretrained_weights(self, pretrained_model: str):
        """update pretrained weight path

        Args:
            pretrained_model (str): the local path or url of pretrained weight file to set.
        """
        assert isinstance(
            pretrained_model, (str, type(None))
        ), "The 'pretrained_model' should be a string, indicating the path to the '*.pdparams' file, or 'None', \
indicating that no pretrained model to be used."

        if pretrained_model is None:
            self.update(['Global.pretrained_model=None'])
            self.update(['Arch.pretrained=False'])
        else:
            if pretrained_model.lower() == "default":
                self.update(['Global.pretrained_model=None'])
                self.update(['Arch.pretrained=True'])
            else:
                if not pretrained_model.startswith(('http://', 'https://')):
                    pretrained_model = abspath(
                        pretrained_model.replace(".pdparams", ""))
                self.update([f'Global.pretrained_model={pretrained_model}'])

    def update_num_classes(self, num_classes: int):
        """update classes number

        Args:
            num_classes (int): the classes number value to set.
        """
        update_str_list = [f'Arch.class_num={num_classes}']
        if self._get_arch_name() == "DistillationModel":
            update_str_list.append(
                f"Arch.models.0.Teacher.class_num={num_classes}")
            update_str_list.append(
                f"Arch.models.1.Student.class_num={num_classes}")
        self.update(update_str_list)

    def _update_slim_config(self, slim_config_path: str):
        """update slim settings

        Args:
            slim_config_path (str): the path to slim config yaml file.
        """
        slim_config = yaml.load(
            open(slim_config_path, 'rb'), Loader=yaml.Loader)['Slim']
        self.update([f'Slim={slim_config}'])

    def _update_amp(self, amp: Union[None, str]):
        """update AMP settings

        Args:
            amp (None | str): the AMP settings.

        Raises:
            ValueError: AMP setting `amp` error, missing field `AMP`.
        """
        if amp is None or amp == 'OFF':
            if 'AMP' in self.dict:
                self._dict.pop('AMP')
        else:
            if 'AMP' not in self.dict:
                raise ValueError("Config must have AMP information.")
            _cfg = ['AMP.use_amp=True', f'AMP.level={amp}']
            self.update(_cfg)

    def update_num_workers(self, num_workers: int):
        """update workers number of train and eval dataloader

        Args:
            num_workers (int): the value of train and eval dataloader workers number to set.
        """
        _cfg = [
            f'DataLoader.Train.loader.num_workers={num_workers}',
            f'DataLoader.Eval.loader.num_workers={num_workers}',
        ]
        self.update(_cfg)

    def update_shared_memory(self, shared_memeory: bool):
        """update shared memory setting of train and eval dataloader
        
        Args:
            shared_memeory (bool): whether or not to use shared memory
        """
        assert isinstance(shared_memeory,
                          bool), "shared_memeory should be a bool"
        _cfg = [
            f'DataLoader.Train.loader.use_shared_memory={shared_memeory}',
            f'DataLoader.Eval.loader.use_shared_memory={shared_memeory}',
        ]
        self.update(_cfg)

    def update_shuffle(self, shuffle: bool):
        """update shuffle setting of train and eval dataloader
        
        Args:
            shuffle (bool): whether or not to shuffle the data
        """
        assert isinstance(shuffle, bool), "shuffle should be a bool"
        _cfg = [
            f'DataLoader.Train.loader.shuffle={shuffle}',
            f'DataLoader.Eval.loader.shuffle={shuffle}',
        ]
        self.update(_cfg)

    def update_dali(self, dali: bool):
        """enable DALI setting of train and eval dataloader
        
        Args:
            dali (bool): whether or not to use DALI
        """
        assert isinstance(dali, bool), "dali should be a bool"
        _cfg = [
            f'Global.use_dali={dali}',
            f'Global.use_dali={dali}',
        ]
        self.update(_cfg)

    def update_seed(self, seed: int):
        """update seed

        Args:
            seed (int): the random seed value to set
        """
        _cfg = [f'Global.seed={seed}']
        self.update(_cfg)

    def update_device(self, device: str):
        """update device setting

        Args:
            device (str): the running device to set
        """
        device = device.split(':')[0]
        _cfg = [f'Global.device={device}']
        self.update(_cfg)

    def update_label_dict_path(self, dict_path: str):
        """update label dict file path

        Args:
            dict_path (str): the path of label dict file to set
        """
        _cfg = [f'PostProcess.Topk.class_id_map_file={abspath(dict_path)}', ]
        self.update(_cfg)

    def _update_to_static(self, dy2st: bool):
        """update config to set dynamic to static mode

        Args:
            dy2st (bool): whether or not to use the dynamic to static mode.
        """
        self.update([f'Global.to_static={dy2st}'])

    def _update_use_vdl(self, use_vdl: bool):
        """update config to set VisualDL

        Args:
            use_vdl (bool): whether or not to use VisualDL.
        """
        self.update([f'Global.use_visualdl={use_vdl}'])

    def _update_epochs(self, epochs: int):
        """update epochs setting

        Args:
            epochs (int): the epochs number value to set
        """
        self.update([f'Global.epochs={epochs}'])

    def _update_checkpoints(self, resume_path: Union[None, str]):
        """update checkpoint setting

        Args:
            resume_path (None | str): the resume training setting. if is `None`, train from scratch, otherwise,
                train from checkpoint file that path is `.pdparams` file.
        """
        if resume_path is not None:
            resume_path = resume_path.replace(".pdparams", "")
        self.update([f'Global.checkpoints={resume_path}'])

    def _update_output_dir(self, save_dir: str):
        """update output directory

        Args:
            save_dir (str): the path to save outputs.
        """
        self.update([f'Global.output_dir={abspath(save_dir)}'])

    def update_log_interval(self, log_interval: int):
        """update log interval(steps)

        Args:
            log_interval (int): the log interval value to set.
        """
        self.update([f'Global.print_batch_step={log_interval}'])

    def update_eval_interval(self, eval_interval: int):
        """update eval interval(epochs)

        Args:
            eval_interval (int): the eval interval value to set.
        """
        self.update([f'Global.eval_interval={eval_interval}'])

    def update_save_interval(self, save_interval: int):
        """update eval interval(epochs)

        Args:
            save_interval (int): the save interval value to set.
        """
        self.update([f'Global.save_interval={save_interval}'])

    def update_log_ranks(self, device):
        """update log ranks

        Args:
            device (str): the running device to set
        """
        log_ranks = device.split(':')[1]
        self.update([f'Global.log_ranks="{log_ranks}"'])

    def update_print_mem_info(self, print_mem_info: bool):
        """setting print memory info"""
        assert isinstance(print_mem_info,
                          bool), "print_mem_info should be a bool"
        self.update([f'Global.print_mem_info={print_mem_info}'])

    def _update_predict_img(self, infer_img: str, infer_list: str=None):
        """update image to be predicted

        Args:
            infer_img (str): the path to image that to be predicted.
            infer_list (str, optional): the path to file that images. Defaults to None.
        """
        if infer_list:
            self.update([f'Infer.infer_list={infer_list}'])
        self.update([f'Infer.infer_imgs={infer_img}'])

    def _update_save_inference_dir(self, save_inference_dir: str):
        """update directory path to save inference model files

        Args:
            save_inference_dir (str): the directory path to set.
        """
        self.update(
            [f'Global.save_inference_dir={abspath(save_inference_dir)}'])

    def _update_inference_model_dir(self, model_dir: str):
        """update inference model directory

        Args:
            model_dir (str): the directory path of inference model fils that used to predict.
        """
        self.update([f'Global.inference_model_dir={abspath(model_dir)}'])

    def _update_infer_img(self, infer_img: str):
        """update path of image that would be predict

        Args:
            infer_img (str): the image path.
        """
        self.update([f'Global.infer_imgs={infer_img}'])

    def _update_infer_device(self, device: str):
        """update the device used in predicting

        Args:
            device (str): the running device setting
        """
        self.update([f'Global.use_gpu={device.split(":")[0]=="gpu"}'])

    def _update_enable_mkldnn(self, enable_mkldnn: bool):
        """update whether to enable MKLDNN

        Args:
            enable_mkldnn (bool): `True` is enable, otherwise is disable.
        """
        self.update([f'Global.enable_mkldnn={enable_mkldnn}'])

    def _update_infer_img_shape(self, img_shape: str):
        """update image cropping shape in the preprocessing

        Args:
            img_shape (str): the shape of cropping in the preprocessing,
                i.e. `PreProcess.transform_ops.1.CropImage.size`.
        """
        self.update([f'PreProcess.transform_ops.1.CropImage.size={img_shape}'])

    def _update_save_predict_result(self, save_dir: str):
        """update directory that save predicting output

        Args:
            save_dir (str): the dicrectory path that save predicting output.
        """
        self.update([f'Infer.save_dir={save_dir}'])

    def update_model(self, **kwargs):
        """update model settings
        """
        for k in kwargs:
            v = kwargs[k]
            self.update([f'Arch.{k}={v}'])

    def update_teacher_model(self, **kwargs):
        """update teacher model settings
        """
        for k in kwargs:
            v = kwargs[k]
            self.update([f'Arch.models.0.Teacher.{k}={v}'])

    def update_student_model(self, **kwargs):
        """update student model settings
        """
        for k in kwargs:
            v = kwargs[k]
            self.update([f'Arch.models.1.Student.{k}={v}'])

    def get_epochs_iters(self) -> int:
        """get epochs

        Returns:
            int: the epochs value, i.e., `Global.epochs` in config.
        """
        return self.dict['Global']['epochs']

    def get_log_interval(self) -> int:
        """get log interval(steps)

        Returns:
            int: the log interval value, i.e., `Global.print_batch_step` in config.
        """
        return self.dict['Global']['print_batch_step']

    def get_eval_interval(self) -> int:
        """get eval interval(epochs)

        Returns:
            int: the eval interval value, i.e., `Global.eval_interval` in config.
        """
        return self.dict['Global']['eval_interval']

    def get_save_interval(self) -> int:
        """get save interval(epochs)

        Returns:
            int: the save interval value, i.e., `Global.save_interval` in config.
        """
        return self.dict['Global']['save_interval']

    def get_learning_rate(self) -> float:
        """get learning rate

        Returns:
            float: the learning rate value, i.e., `Optimizer.lr.learning_rate` in config.
        """
        return self.dict['Optimizer']['lr']['learning_rate']

    def get_warmup_epochs(self) -> int:
        """get warmup epochs

        Returns:
            int: the warmup epochs value, i.e., `Optimizer.lr.warmup_epochs` in config.
        """
        return self.dict['Optimizer']['lr']['warmup_epoch']

    def get_label_dict_path(self) -> str:
        """get label dict file path

        Returns:
            str: the label dict file path, i.e., `PostProcess.Topk.class_id_map_file` in config.
        """
        return self.dict['PostProcess']['Topk']['class_id_map_file']

    def get_batch_size(self, mode='train') -> int:
        """get batch size

        Args:
            mode (str, optional): the mode that to be get batch size value, must be one of 'train', 'eval', 'test'.
                Defaults to 'train'.

        Returns:
            int: the batch size value of `mode`, i.e., `DataLoader.{mode}.sampler.batch_size` in config.
        """
        return self.dict['DataLoader']['Train']['sampler']['batch_size']

    def get_qat_epochs_iters(self) -> int:
        """get qat epochs

        Returns:
            int: the epochs value.
        """
        return self.get_epochs_iters()

    def get_qat_learning_rate(self) -> float:
        """get qat learning rate

        Returns:
            float: the learning rate value.
        """
        return self.get_learning_rate()

    def _get_arch_name(self) -> str:
        """get architecture name of model

        Returns:
            str: the model arch name, i.e., `Arch.name` in config.
        """
        return self.dict["Arch"]["name"]

    def _get_dataset_root(self) -> str:
        """get root directory of dataset, i.e. `DataLoader.Train.dataset.image_root`

        Returns:
            str: the root directory of dataset
        """
        return self.dict["DataLoader"]["Train"]['dataset']['image_root']

    def get_train_save_dir(self) -> str:
        """get the directory to save output

        Returns:
            str: the directory to save output
        """
        return self['Global']['output_dir']