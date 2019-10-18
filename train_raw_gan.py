import matplotlib.pylab as plt
import numpy as np
import os.path as osp
import random
import torch
import torch.autograd as autograd
from torch.nn.functional import leaky_relu
import torch.optim

from src.autoencoder import Configuration as Conf
from src.datasets import ShapenetCore
from src.neural_net import MODEL_SAVER_ID
from src.in_out import snc_category_to_synth_id, create_dir, files_in_subdirs
from src.general_utils import plot_3d_point_cloud
# from src.vanilla_gan import Vanilla_GAN
from src.w_gan_gp import W_GAN_GP
from src.generators_discriminators import point_cloud_generator, mlp_discriminator


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1) # random uniform
    alpha = alpha.expand(real_samples.size())
    if torch.cuda.is_available():
        alpha = alpha.cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = torch.ones(batch_size, 1)
    if torch.cuda.is_available():
        fake = fake.cuda()
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sample_noise(batch_size, noise_params):
    z = torch.randn(batch_size, noise_params['noise_dim'])
    z = z * noise_params['sigma'] + noise_params['mu']
    if torch.cuda.is_available():
        z = z.cuda()
    return z


def train(train_loader, discriminator, generator, optD, optG, epoch, noise_params, discriminator_boost=5):
    generator.train()
    discriminator.train()
    train_loss = []

    lambda_gp = 10.

    epoch_loss_d = []
    epoch_loss_g = []

    for batch_idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]

        if torch.cuda.is_available():
            data = data.cuda()

        # train discriminator
        optD.zero_grad()
        # z = model.sample_noise(batch_size, noise_params)
        z = sample_noise(batch_size, noise_params)
        fake_data = generator(z)

        # l_D = model.loss_D(data, z)
        real_logit = discriminator(data)
        fake_logit = discriminator(fake_data)
        gradient_penalty = compute_gradient_penalty(discriminator, data.data, fake_data.data)

        d_loss = -torch.mean(real_logit) + torch.mean(fake_logit) + lambda_gp * gradient_penalty

        d_loss.backward()
        optD.step()

        epoch_loss_d.append(d_loss.detach())
        # print('d:', float(d_loss))

        # train generator
        if (batch_idx + 1) % discriminator_boost == 0:
            optG.zero_grad()
            # z = torch.randn(batch_size, noise_params['noise_dim'])
            # z = z * noise_params['sigma'] + noise_params['mu']
            z = sample_noise(batch_size, noise_params)
            # l_G = model.loss_G(z)
            fake_data = generator(z)
            fake_logit = discriminator(fake_data)
            g_loss = -torch.mean(fake_logit)
            g_loss.backward()
            optG.step()

            epoch_loss_g.append(g_loss.detach())
            # print(float(g_loss))

    epoch_loss_d = torch.mean(torch.tensor(epoch_loss_d))
    epoch_loss_g = torch.mean(torch.tensor(epoch_loss_g))
    print("epoch %d: d-loss(%f), g-loss(%f)" % (epoch, epoch_loss_d, epoch_loss_g))
    return epoch_loss_d, epoch_loss_g


if __name__ == "__main__":
    # Use to save Neural-Net check-points etc.
    top_out_dir = './data/'

    # Top-dir of where point-clouds are stored.
    top_in_dir = './data/shape_net_core_uniform_samples_2048/'

    experiment_name = 'raw_gan_with_w_gan_loss'

    n_pc_points = 2048 # Number of points per model.
    # class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
    class_name = 'chair'

    use_wgan = True     # Wasserstein with gradient penalty, or not?
    n_epochs = 50       # Epochs to train.

    plot_train_curve = True
    save_gan_model = True
    saver_step = np.hstack([np.array([1, 5, 10]), np.arange(50, n_epochs + 1, 50)])

    # If true, every 'saver_step' epochs we produce & save synthetic pointclouds.
    save_synthetic_samples = True
    # How many synthetic samples to produce at each save step.
    n_syn_samples = 10 # all_pc_data.num_examples

    # Optimization parameters
    init_lr = 0.0001
    batch_size = 50
    noise_dim = 128
    noise_params = {'noise_dim': noise_dim, 'mu':0, 'sigma': 0.2}
    betas = (0.5, 0.9) # ADAM's momentum.

    # Load point-clouds.
    syn_id = snc_category_to_synth_id()[class_name]
    class_dir = osp.join(top_in_dir , syn_id)
    file_names = [f for f in files_in_subdirs(class_dir, '.ply')]
    random.shuffle(file_names)
    total_num = len(file_names)
    train_loader = torch.utils.data.DataLoader(ShapenetCore(file_names), batch_size=50, shuffle=True)

    n_out = [3, n_pc_points] # Dimensionality of generated samples.

    # discriminator = mlp_discriminator
    # generator = point_cloud_generator

    """
    if save_synthetic_samples:
        synthetic_data_out_dir = osp.join(top_out_dir, 'OUT/synthetic_samples/', experiment_name)
        create_dir(synthetic_data_out_dir)

    if save_gan_model:
        train_dir = osp.join(top_out_dir, 'OUT/raw_gan', experiment_name)
        create_dir(train_dir)
    """
    train_dir = osp.join(top_out_dir, 'raw_gan')
    create_dir(train_dir)

    if use_wgan:
        lam = 10
        disc_kwargs = {'b_norm': False}

        # gan = W_GAN_GP(lam, n_out, noise_dim, discriminator, generator, disc_kwargs=disc_kwargs)
        generator = point_cloud_generator(noise_dim, n_out)
        discriminator = mlp_discriminator(**disc_kwargs)
    else:
        raise NotImplementedError("Vanilla_GAN not implemented")
        """
        leak = 0.2
        disc_kwargs = {'non_linearity': leaky_relu(leak), 'b_norm': False}
        gan = Vanilla_GAN(experiment_name, init_lr, n_out, noise_dim,
                          discriminator, generator, beta=beta, disc_kwargs=disc_kwargs)
        """

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    accum_syn_data = []
    train_stats = []

    # Train the GAN.
    opt_d = torch.optim.Adam(discriminator.parameters(), init_lr, betas=betas)
    opt_g = torch.optim.Adam(generator.parameters(), init_lr, betas=betas)
    # .minimize(loss, var_list=var_list)

    for epoch in range(n_epochs):
        train(train_loader, discriminator, generator, opt_d, opt_g, epoch, noise_params)
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), osp.join(train_dir, "model%d.pth" % epoch))
    exit()

    if plot_train_curve:
        x = range(len(train_stats))
        d_loss = [t[1] for t in train_stats]
        g_loss = [t[2] for t in train_stats]
        plt.plot(x, d_loss, '--')
        plt.plot(x, g_loss)
        plt.title('GAN training. (%s)' %(class_name))
        plt.legend(['Discriminator', 'Generator'], loc=0)
        
        plt.tick_params(axis='x', which='both', bottom='off', top='off')
        plt.tick_params(axis='y', which='both', left='off', right='off')
        
        plt.xlabel('Epochs.') 
        plt.ylabel('Loss.')
