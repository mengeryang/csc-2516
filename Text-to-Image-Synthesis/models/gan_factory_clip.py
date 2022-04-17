from models import gan, gan_cls, wgan_cls, wgan

class gan_factory(object):
        

    @staticmethod
    def generator_factory(type, embed_dim):
        if type == 'gan':
            return gan_cls.generator(embed_dim=embed_dim)
        elif type == 'wgan':
            return wgan_cls.generator(embed_dim=embed_dim)
        elif type == 'vanilla_gan':
            return gan.generator()
        elif type == 'vanilla_wgan':
            return wgan.generator()

    @staticmethod
    def discriminator_factory(type, embed_dim):
        if type == 'gan':
            return gan_cls.discriminator(embed_dim=embed_dim)
        elif type == 'wgan':
            return wgan_cls.discriminator(embed_dim=embed_dim)
        elif type == 'vanilla_gan':
            return gan.discriminator()
        elif type == 'vanilla_wgan':
            return wgan.discriminator()
