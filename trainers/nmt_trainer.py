import os

from base.trainer import BaseTrainer


class NMTTrainer(BaseTrainer):

    def __init__(self, model, optimizer, criterion,  metric, device, lr_scheduler = None, log = None):
        super(NMTTrainer,self).__init__(model,optimizer,criterion,metric,device,lr_scheduler,log)


    def iter(self, batch, is_train = True):

        src = batch[0]
        trg = batch[1]

        src = src.to(self.device)
        trg = trg.to(self.device)

        output, _ = self.model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = self.criterion(output, trg)

        return loss, output, trg
