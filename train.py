import torch
import os
import numpy as np
from tqdm import tqdm
import sys
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.data import GBufferDataset, ImageDataset, prepare_data
from src.model import ToonShaderStyle, VGGFeatureExtractor
from src.loss import smooth_loss
from src.fileIO import safetensorFromCkpt

def parsingConf(conf : dict) :
    device = conf['device'] if conf['device'] is not None else 'cpu'
    networks_conf = conf.get('networks')
    networks = {}
    if networks_conf is not None :
        for k, v in networks_conf.items() :
            
            if v['name'] == 'ToonShaderStyle' :
                networks[k] = ToonShaderStyle(
                    negative_slope=v['negative_slope'] if v.get('negative_slope') is not None else 0.01,
                    weight_norm=v['weight_norm'] if v.get('weight_norm') is not None else False
                ).to(device)

            elif v['name'] == 'VGGFeatureExtractor' :
                networks[k] = VGGFeatureExtractor().to(device)

            else :
                assert False, 'no matching key for network'

            if v.get('pretrained') is not None :
                networks[k].load_state_dict(torch.load(v['pretrained']['path'])[v['pretrained']['key']])
                print(f"pretrained networks : {v['pretrained']['path']} is loaded")

    dataloader_conf = conf.get('dataloaders')
    dataloaders = {}
    if dataloader_conf is not None :
        for k, v in dataloader_conf.items() :
            if v['name'] == 'GBufferDataset' :
                dataset = GBufferDataset(v['directory'])
            elif v['name'] == 'ImageDataset' :
                dataset = ImageDataset(v['directory'])
            else :
                assert False, 'no matching  key for dataloader'

            if v.get('random_split') is not None :
                train_size = int(len(dataset) * v['random_split'])
                val_size = len(dataset) - train_size
                train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
                dataloaders[k + '_train'] = DataLoader(train_set, v['batch_size'], True, num_workers=0, drop_last=False)
                dataloaders[k + '_val'] = DataLoader(val_set, v['batch_size'], True, num_workers=0, drop_last=False)
            else :
                dataloaders[k] = DataLoader(dataset, v['batch_size'], True, num_workers=0, drop_last=False)

    optimizer_conf = conf.get('optimizers')
    optimizers = {}
    if optimizer_conf is not None :
        for k, v in optimizer_conf.items() :
            lr = v['lr'] if v.get('lr') is not None else 1e-3
            weight_decay = v['weight_decay'] if v.get('weight_decay') is not None else 0.
            if v['name'] == 'Adam' :
                optimizers[k] = torch.optim.Adam(networks[k].parameters(), lr=lr, weight_decay=weight_decay)
            elif v['name'] == 'RAdam' :
                optimizers[k] = torch.optim.RAdam(networks[k].parameters(), lr=lr, weight_decay=weight_decay)
            elif v['name'] == 'AdamW' :
                optimizers[k] = torch.optim.AdamW(networks[k].parameters(), lr=lr, weight_decay=weight_decay)
            else :
                assert False, 'no matching key for optimizer'

    scheduler_conf = conf.get('schedulers')
    schedulers = {}
    if scheduler_conf is not None :
        for k, v in scheduler_conf.items() :
            if v['name'] == 'ExponentialLR' :
                schedulers[k] = torch.optim.lr_scheduler.ExponentialLR(optimizers[k], v['gamma'])
            elif v['name'] == 'Warmup' :
                lr_decay = v['lr_decay'] if v.get('lr_decay') is not None else 1.
                warmup_epoch = v['warmup_epoch'] if v.get('warmup_epoch') is not None else 0
                schedulers[k] = torch.optim.lr_scheduler.LambdaLR(optimizers[k], lr_lambda=lambda epoch : 0. if epoch < warmup_epoch else lr_decay**(epoch-warmup_epoch))
            else :
                assert False, 'no matching key for scheduler'

    return networks, dataloaders, optimizers, schedulers, device

class TrainPipeline() :
    def __init__(self, conf) -> None:
        self.networks, self.dataloaders, self.optimizers, self.schedulers, self.device = parsingConf(conf)

    def learning_pipeline(self, isTraining=False) :
        return {'scalar' : None, 'image' : None}

    def update_scheduler(self) -> None:
        if self.schedulers is None :
            return

        elif isinstance(self.schedulers, dict) :
            for key, scheduler in self.schedulers.items() :
                scheduler.step()
        else :
            self.schedulers.step()

    def train(self) :
        return self.learning_pipeline(True)

    def validation(self) :
        return self.learning_pipeline(False)
    
# ##================================================================================================================================
#  ToonOnly pipeline
# ##================================================================================================================================

class ToonOnlyPipeline(TrainPipeline) :
    DEFAULT_CONF = {
        'device' : 'cuda',
        'epoch' : 200,
        'save_period' : 50,

        'networks' : {
            'net' : {
                'name' : 'ToonShaderStyle',
                'weight_norm' : True
            }
        },
        'dataloaders' : {
            'net' : {
                'name' : 'GBufferDataset',
                'directory' : 'dataset/gbuffer/',
                'batch_size' : 4,
                'random_split' : 11/12
            }
        },
        'optimizers' : {
            'net' : {
                'name' : 'RAdam',
                'lr' : 1e-3,
                'weight_decay' : 1e-5
            }
        },
        'schedulers' : {
            'net' : {
                'name' : 'ExponentialLR',
                'gamma' : 0.99
            }
        }
    }

    def __init__(self, conf) -> None:
        super(ToonOnlyPipeline, self).__init__(conf)

    def learning_pipeline(self, isTraining=False):
        dataloader = self.dataloaders['net_train'] if isTraining else self.dataloaders['net_val']
        loss_arr = []

        net = self.networks['net']
        optim = self.optimizers['net']
        if isTraining :
            net.train()
        else :
            net.eval()

        for _, data in enumerate(tqdm(dataloader)) :
            g_buffer, img = prepare_data(data, self.device)

            output = net(g_buffer)
            divisor = torch.sum(img[:, 3:])
            loss_rgb = torch.nn.MSELoss(reduction='sum')(output[:, :3] * img[:, 3:], img[:, :3]*img[:, 3:]) / divisor
            loss = loss_rgb

            if isTraining :
                optim.zero_grad()
                loss.backward()
                optim.step()

            loss_arr.append([loss_rgb.detach().item()])

        loss_mean = np.mean(loss_arr, axis=0)

        return {'scalar' : {'loss_rgb' : loss_mean[0]},
                'images' : {'target' : img[:, :3].cpu().detach(), 'output' : output[:, :3].cpu().detach()}}
    
class StyleTransferPipeline(TrainPipeline) :
    DEFAULT_CONF = {
        'device' : 'cuda', 
        'epoch' : 400,
        'save_period' : 100,
        
        'pipeline' : {
            'loss_rgb' : 8,
            'loss_style' : 1,
            'loss_smooth' : 11,
        },
        'networks' : {
            'net' : {
                'name' : 'ToonShaderStyle',
                'weight_norm' : True,
                'pretrained' :{
                    'path' : None,
                    'key' : 'net'
                }
            },
            'FE' : {
                'name' : 'VGGFeatureExtractor'
            }
        },
        'dataloaders' : {
            'S' : {
                'name' : 'GBufferDataset',
                'directory' : 'dataset/gbuffer/',
                'batch_size' : 4,
                'random_split' : 11/12
            },
            'R' : {
                'name' : 'ImageDataset',
                'directory' : 'dataset/style/',
                'batch_size' : 4,
                'random_split' : 11/12
            }
        },
        'optimizers' : {
            'net' : {
                'name' : 'RAdam',
                'lr' : 4e-4,
                'weight_decay' : 1e-5
            }
        },
        'schedulers' : {
            'net' : {
                'name' : 'ExponentialLR',
                'gamma' : 0.99999
            }
        }
    }
    
    def __init__(self, conf) -> None:
        super(StyleTransferPipeline, self).__init__(conf)
        pipeline_conf = conf['pipeline'] if conf.get('pipeline') is not None else {}
        self.const = {
            'loss_style' : 10,
            'loss_rgb' : 10,
            'loss_smooth' : 10,
        }
        for k, v in pipeline_conf.items() : 
            self.const[k] = v

    def get_feature_vector(self, features:dict) :
        mean_std = [self.get_feature_mean_std(v) for v in features.values()]
        feature_vector = [v[0] for v in mean_std] + [v[1] for v in mean_std]
        return torch.concat(feature_vector, dim=1)
    
    def get_feature_mean_std(self, feature:torch.Tensor) :
        return torch.mean(feature, dim=(-1, -2)), torch.std(feature, dim=(-1, -2))
    
    def style_loss(self, input_features:dict, target_features:dict) :
        layers = ['relu1_1', 'relu2_1', 'relu3_1']
        loss = 0.
        for layer in layers : 
            input_mean, input_std = self.get_feature_mean_std(input_features[layer])
            target_mean, target_std = self.get_feature_mean_std(target_features[layer])
            loss += torch.nn.MSELoss()(input_mean, target_mean.detach())
            loss += torch.nn.MSELoss()(input_std, target_std.detach()) 
        return loss 
        
    def content_loss(self, input_features:dict, target_features:dict) :
        input_mean, input_std = self.get_feature_mean_std(input_features['relu4_1'])
        target_mean, target_std = self.get_feature_mean_std(target_features['relu4_1'])
        input_norm = (input_features['relu4_1'] - input_mean[..., None, None]) / (input_std[..., None, None] + 1e-8)
        target_norm = (target_features['relu4_1'] - target_mean[..., None, None]) / (target_std[..., None, None] + 1e-8)
        return torch.nn.MSELoss()(input_norm, target_norm.detach())

    def learning_pipeline(self, isTraining=False):
        dataloader_S = self.dataloaders['S_train'] if isTraining else self.dataloaders['S_val']
        dataloader_R = self.dataloaders['R_train'] if isTraining else self.dataloaders['R_val']
        
        loss_arr = []
        s_iter = iter(dataloader_S)
        net = self.networks['net']
        net_FE = self.networks['FE']
        optim = self.optimizers['net']

        net_FE.eval()
        if not isTraining :
            net.eval()

        for _, data_r in enumerate(tqdm(dataloader_R)) :
            try : 
                data_s = next(s_iter)
                g_buffer_s, img_s = prepare_data(data_s, self.device)

                img_s_alpha = img_s[:, 3:]
                img_s = img_s[:, :3]
            except StopIteration:
                break
            img_r = data_r['image'].squeeze(0).to('cuda')[:, :3]
            
            # ==================================================================================
            # ++++++++++++++++++++++++++ style transfer back-prop -------------------------------
            
            output = net(g_buffer_s)[:, :3]
            divisor = torch.sum(img_s_alpha)
            loss_rgb = torch.nn.MSELoss(reduction='sum')(output*img_s_alpha, img_s*img_s_alpha) / divisor

            target_features = net_FE(img_r)
            origin_features = net_FE(output.detach())

            feature_vector = self.get_feature_vector(target_features)
            output_styled = net(g_buffer_s, feature_vector)[:, :3]

            output_features = net_FE(output_styled)
            loss_c = self.content_loss(output_features, origin_features)
            loss_s = self.style_loss(output_features, target_features) 
            loss_r = smooth_loss(output_styled, img_r, self.const['loss_smooth'])

            loss = self.const['loss_rgb']*loss_rgb + self.const['loss_style']*loss_s + loss_c + loss_r
            
            if isTraining :
                net.train()

                optim.zero_grad()
                loss.backward()
                optim.step()
                
            loss_arr.append([loss_rgb.detach().item(), loss_c.detach().item(), loss_s.detach().item(), loss_r.detach().item()])
        
        loss_mean = np.mean(loss_arr, axis=0)     
        return {
            'scalar' : {
                'loss_rgb' : loss_mean[0],  'loss_c' : loss_mean[1], 'loss_s' : loss_mean[2], 'loss_r' : loss_mean[3]
            },
            'image' : {
                'target_s' : img_s[0].cpu().detach(),
                'target_r' : img_r[0].cpu().detach(), 
                'output' : output[0].cpu().detach(),
                'output_r' : output_styled[0].cpu().detach()
            }
        }
    
# ================================================================================================================================
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#  Train Logger
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ================================================================================================================================
class TrainLogger() :
    def __init__(self, log_dir, save_period:int=10) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_period = save_period

        self.ckpt_dir = os.path.join(log_dir, 'Checkpoint', timestamp)
        self.log_dir = os.path.join(log_dir, 'log', timestamp)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer_train = SummaryWriter(log_dir=os.path.join(self.log_dir, 'train'))
        self.writer_val = SummaryWriter(log_dir=os.path.join(self.log_dir, 'val'))

    def logging(self, writer:SummaryWriter, metrics:dict, epoch:int) -> None:
        if metrics.get('scalar') is not None :
            for k, v in metrics['scalar'].items() :
                writer.add_scalar(k, v, epoch)

        if metrics.get('image') is not None :
            for k, v in metrics['image'].items() :
                writer.add_image(k, v, epoch, dataformats='CHW')

        if metrics.get('images') is not None :
            for k, v in metrics['images'].items() :
                writer.add_images(k, v, epoch, dataformats='NCHW')

    def log_train(self, metrics, epoch) -> None:
        self.logging(self.writer_train, metrics, epoch)

    def log_val(self, metrics, epoch) -> None:
        self.logging(self.writer_val, metrics, epoch)

    def print_log(self, metrics, epoch) -> None:
        log_txt = f'validation : epoch {epoch} |'
        for k, v in metrics['scalar'].items() :
            log_txt += f' {k} {v:.4f} |'
        print(log_txt)

    def save_networks(self, networks, optims, schedulers, epoch:int) :
        if epoch % self.save_period == 0 : 
            save_dict = {
                'net' : networks['net'].state_dict(),
                'optim' : optims['net'].state_dict(),
                'scheduler' : schedulers['net'].state_dict(),
                'epoch' : epoch
            }
            torch.save(save_dict, f'{self.ckpt_dir}/ckpt_epoch{epoch:03}.tar')

def trainer_iteration(pipeline : TrainPipeline, logger:TrainLogger, max_epoch=200) :
    for epoch in range(1, max_epoch+1) :

        # train
        metrics = pipeline.train()
        logger.log_train(metrics, epoch)

        # validation
        with torch.no_grad() :
            metrics = pipeline.validation()
            logger.log_val(metrics, epoch)
            logger.print_log(metrics, epoch)

        # scheduler update
        pipeline.update_scheduler()

        logger.save_networks(pipeline.networks, pipeline.optimizers, pipeline.schedulers, epoch)

if __name__ == "__main__" :
    
    # prepare trainig conf file
    if len(sys.argv) < 2 :
        conf = {
            'stage1' : ToonOnlyPipeline.DEFAULT_CONF,
            'stage2' : StyleTransferPipeline.DEFAULT_CONF,
            'export' : 'model/pretrained.safetensors'
        }

    else :
        with open(sys.argv[1], 'r') as file :
            conf = yaml.safe_load(file)

    # stage 1 : training toon shader only
    if conf.get('stage1') is not None : 
        pipeline = ToonOnlyPipeline(conf['stage1'])
        logger = TrainLogger('training/', save_period=conf['stage1']['save_period'])
        trainer_iteration(pipeline, logger, conf['stage1']['epoch'])

    # stage 2 : training toon shader with style 
    if conf.get('stage2') is not None : 
        if conf['stage2']['networks']['net']['pretrained']['path'] is None : 
            last_ckpt = sorted(os.listdir(logger.ckpt_dir), key=lambda x:x[-7:-4])[-1]
            conf['stage2']['networks']['net']['pretrained']['path'] = os.path.join(logger.ckpt_dir, last_ckpt)

        pipeline = StyleTransferPipeline(conf['stage2'])
        logger = TrainLogger('training/', save_period=conf['stage2']['save_period'])
        trainer_iteration(pipeline, logger, conf['stage2']['epoch'])

    # save network as safetensors
    if conf.get('export') is not None :
        last_ckpt = sorted(os.listdir(logger.ckpt_dir), key=lambda x:x[-7:-4])[-1]
        safetensorFromCkpt(pipeline.networks['net'], os.path.join(logger.ckpt_dir, last_ckpt), conf['export'])


