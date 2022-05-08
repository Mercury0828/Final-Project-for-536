r"""
CUDA_VISIBLE_DEVICES=0 python main.py --dataset i_raven --validate_interval 1 --verbose 3 --color --batch_size 100 --lr 1e-3 --epochs 300 --configure left_center_single_right_center_single --tensorboard --model cnn

--amp
--tqdm

""" # noqa

import trojanvision
from trojanzoo.utils.model import init_weights
import torch
import argparse


from dataset import IRaven
from model import CNN, ResNet, LSTM, WReN, EfficientNet


if __name__ == '__main__':
    trojanvision.datasets.class_dict['i_raven'] = IRaven
    trojanvision.models.class_dict['cnn'] = CNN
    trojanvision.models.class_dict['resnet'] = ResNet
    trojanvision.models.class_dict['lstm'] = LSTM
    trojanvision.models.class_dict['wren'] = WReN
    trojanvision.models.class_dict['efficientnet'] = EfficientNet
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    parser.add_argument('--configure', default='center_single')
    # arg_list = ['--dataset', 'i_raven',
    #             '--model', 'resnet',
    #             '--validate_interval', '1',
    #             '--verbose', '3',
    #             '--color', '--tqdm',
    #             '--batch_size', '100',
    #             '--lr', '1e-4',
    #             '--epochs', '300',
    #             '--amp',
    #             # '--tensorboard', '--log_dir', './tensorboard/',
    #             '--lr_scheduler']
    # kwargs = parser.parse_args(args=arg_list).__dict__
    kwargs = parser.parse_args().__dict__
    kwargs['log_dir'] = f'./tensorboard/{kwargs["configure"]}/{kwargs["model_name"]}'

    env = trojanvision.environ.create(**kwargs)
    model_name = kwargs['model_name']
    if ('resnet' not in model_name and 'efficientnet' not in model_name) \
            or 'comp' in model_name:
        data_shape = [16, 80, 80]
    else:
        data_shape = [16, 224, 224]
    dataset: IRaven = trojanvision.datasets.create(
        folder_path='./Dataset', data_shape=data_shape, **kwargs)

    model = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset,
                                          model=model, **kwargs)
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'],
                                         betas=(0.9, 0.999), eps=1e-8)
    trainer.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=trainer['epochs'])

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset,
                             model=model,
                             trainer=trainer)

    init_weights(model._model)
    model._train(**trainer)
