import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time

from models.gan_factory_clip import gan_factory
from PIL import Image
import os
from models.clip_encoder import get_clip_txt_embeddings
from models.bert_encoder import get_bert_txt_embeddings
class GANModel(object):
    def __init__(self, name, type, encoder_type, pre_trained_gen, pre_trained_disc, embed_dim=512):
        self.name = name
        self.generator = torch.nn.DataParallel(gan_factory.generator_factory(type, embed_dim).cuda())
        self.discriminator = torch.nn.DataParallel(gan_factory.discriminator_factory(type, embed_dim).cuda())

        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc))
        else:
            print('no pretrained disc model given !')
            assert(1==2)

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen))
        else:
            print('no pretrained gen model given !')
            assert(1==2)

        self.noise_dim = 100

        self.type = type
        self.encoder_type = encoder_type
        self.generator.eval()
        self.discriminator.eval()

    def sample_inference(self, txtlist):
        '''
        txtlist: list of text
        '''
        if self.encoder_type == 'clip':
            txt_embeds = get_clip_txt_embeddings(txtlist)
        elif self.encoder_type == 'bert':
            txt_embeds = get_bert_txt_embeddings(txtlist)

        txt_embeds = Variable(torch.FloatTensor(txt_embeds.astype(float))).cuda()
        noise = Variable(torch.randn(txt_embeds.size(0), 100)).cuda()
        noise = noise.view(noise.size(0), 100, 1, 1)
        print('shape ', txt_embeds.shape, noise.shape, '--------')
        fake_images = self.generator(txt_embeds, noise)

        
        if not os.path.exists('inference_result/{0}'.format(self.name)):
            os.makedirs('inference_result/{0}'.format(self.name))


        for t, image in zip(txtlist, fake_images):
            im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            im.save('inference_result/{0}/{1}.jpg'.format(self.name, t.replace("/", "")[:100]))
            print(t)

config_list = [
    # bert
    {
        'name': 'bert',
        'type': 'gan',
        'encoder_type': 'bert', 
        'pre_trained_gen': 'checkpoints_' + 'bert' + '/gen_190.pth', 
        'pre_trained_disc': 'checkpoints_' + 'bert' + '/disc_190.pth',  
        'embed_dim': 768
    },
    # bert_cls
    {
        'name': 'bert_cls',
        'type': 'gan',
        'encoder_type': 'bert', 
        'pre_trained_gen': 'checkpoints_' + 'bert_cls' + '/gen_190.pth', 
        'pre_trained_disc': 'checkpoints_' + 'bert_cls' + '/disc_190.pth',  
        'embed_dim': 768
    },
    # bert_cls_int
    {
        'name': 'bert_cls_int',
        'type': 'gan',
        'encoder_type': 'bert', 
        'pre_trained_gen': 'checkpoints_' + 'bert_cls_int' + '/gen_190.pth', 
        'pre_trained_disc': 'checkpoints_' + 'bert_cls_int' + '/disc_190.pth',  
        'embed_dim': 768
    },
    
    # clip
    {
        'name': 'clip',
        'type': 'gan',
        'encoder_type': 'clip', 
        'pre_trained_gen': 'checkpoints_' + 'clip' + '/gen_190.pth', 
        'pre_trained_disc': 'checkpoints_' + 'clip' + '/disc_190.pth',  
        'embed_dim': 512
    },
    # clip_cls
    {
        'name': 'clip_cls',
        'type': 'gan',
        'encoder_type': 'clip', 
        'pre_trained_gen': 'checkpoints_' + 'clip_cls' + '/gen_190.pth', 
        'pre_trained_disc': 'checkpoints_' + 'clip_cls' + '/disc_190.pth',  
        'embed_dim': 512
    },
    # clip_cls_int
    {
        'name': 'clip_cls_int',
        'type': 'gan',
        'encoder_type': 'clip', 
        'pre_trained_gen': 'checkpoints_' + 'clip_cls_int' + '/gen_190.pth', 
        'pre_trained_disc': 'checkpoints_' + 'clip_cls_int' + '/disc_190.pth',  
        'embed_dim': 512
    },
]

model_list = []
for config in config_list:
    try:
        model = GANModel(**config)
        model_list.append(model)
    except:
        print(config, 'wrong')

print('model initialize done')
txtlist = ['a big grey and white bird with a white underbelly and long wings.']

for model in model_list:
    model.sample_inference(txtlist)



