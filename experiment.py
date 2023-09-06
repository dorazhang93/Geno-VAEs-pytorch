import os
import math
from torch import optim
from models import BaseVAE
from models.types_ import *
import pytorch_lightning as pl
import json
import torch


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.metrics = self.metrics_setup()
        # self.example_input_array = torch.rand((1,self.model.in_dim))
        self.validation_step_outputs = []
        # use manual optimization for mix-precision training
        self.automatic_optimization =False
        self.scaler = torch.cuda.amp.GradScaler()

    def metrics_setup(self):
        inf=math.inf
        return {'val': {'loss': inf, 'Reconstruction_loss':inf, 'Reg_loss':inf}}

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            genotypes = batch
            self.curr_device = genotypes[0].device if isinstance(genotypes, list) else genotypes.device
            results = self.forward(genotypes)

            train_loss = self.model.loss_function(*results,
                                                optimizer_idx=optimizer_idx,
                                                batch_idx = batch_idx)
            loss = train_loss['loss']
        self.manual_backward(self.scaler.scale(loss))
        opt = self.optimizers()
        self.scaler.step(opt)
        self.scaler.update()
        opt.zero_grad()

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return loss
    def on_train_epoch_end(self) -> None:
        # update LR scheduler every epoch
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        genotypes = batch
        self.curr_device = genotypes[0].device if isinstance(genotypes, list) else genotypes.device

        results = self.forward(genotypes)

        val_loss = self.model.loss_function(*results,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.validation_step_outputs.append(val_loss)
        return val_loss

    def on_validation_epoch_end(self,):
        outputs = self.validation_step_outputs
        val_loss={'loss': 0, 'Reconstruction_loss':0, 'Reg_loss':0}
        steps=0
        for output in outputs:
            steps+=1
            val_loss['loss']+=output['loss'].cpu().numpy()
            val_loss['Reconstruction_loss']+=output['Reconstruction_loss'].cpu().numpy()
            val_loss['Reg_loss']+=output['Reg_loss'].cpu().numpy()
        val_loss={key: val/steps for key,val in val_loss.items()}
        print("steps %d" % steps, val_loss)
        if val_loss['loss']< self.metrics['val']['loss']:
            self.metrics['val']=val_loss
            with open(os.path.join(self.logger.log_dir,"metrics.json"),"w") as outfile:
                outfile.write(json.dumps(self.metrics))
        self.validation_step_outputs.clear()

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                        gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
