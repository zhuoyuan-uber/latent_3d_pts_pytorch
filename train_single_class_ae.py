import os.path as osp
import random
import torch.optim

from src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from src.autoencoder import Configuration as Conf
from src.datasets import ShapenetCore
from src.general_utils import plot_3d_point_cloud
from src.point_net_ae import PointNetAutoEncoder
from src.in_out import snc_category_to_synth_id, create_dir, files_in_subdirs
# from src.tf_utils import reset_tf_graph
# from src.general_utils import plot_3d_point_cloud

def train(train_loader, model, optimizer, epoch, args):
    model.train()
    train_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        x_reconstr = model(data)
        l = model.loss(x_reconstr, target)
        l.backward()
        optimizer.step()
        train_loss.append(float(l))
    loss_avr = torch.mean(torch.tensor(train_loss))
    print("Epoch: %d, train loss: %f" % (epoch, loss_avr))


def test(test_loader, model, args):
    print('testing...')
    model.eval()
    test_loss = []
    for batch_idx, (data, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        x_reconstr = model(data)
        l = model.loss(x_reconstr, target)
        test_loss.append(float(l))
    test_loss = torch.tensor(test_loss)
    loss_avr = torch.mean(test_loss)
    print("Test loss: %f" % loss_avr)
    # randomly visualize a few
    for i in range(3):
        xyz = x_reconstr[i,:,:].detach()
        xyz = xyz.cpu().numpy()
        visualize(xyz)


def visualize(x_reconstr):
    plot_3d_point_cloud(x_reconstr[0,:], x_reconstr[1,:], x_reconstr[2,:], in_u_sphere=True)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    top_out_dir = './data/'          # Use to save Neural-Net check-points etc.
    top_in_dir = './data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

    experiment_name = 'single_class_ae'
    n_pc_points = 2048                # Number of points per model.
    bneck_size = 128                  # Bottleneck-AE size
    ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'
    # class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
    class_name = "chair"

    train_params = default_train_params()

    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
    train_dir = create_dir(osp.join(top_out_dir, experiment_name))

    args = Conf(n_input=[3, n_pc_points],
                loss=ae_loss,
                training_epochs = train_params['training_epochs'],
                batch_size = train_params['batch_size'],
                denoising = train_params['denoising'],
                learning_rate = train_params['learning_rate'],
                weight_decay=0,
                train_dir=train_dir,
                loss_display_step = train_params['loss_display_step'],
                saver_step = train_params['saver_step'],
                encoder=encoder,
                decoder=decoder,
                encoder_args=enc_args,
                decoder_args=dec_args
               )

    args.experiment_name = experiment_name
    args.held_out_step = 5   # How often to evaluate/print out loss on 
                             # held_out data (if they are provided in ae.train() ).
    args.save(osp.join(train_dir, 'configuration'))

    syn_id = snc_category_to_synth_id()[class_name]
    class_dir = osp.join(top_in_dir , syn_id)
    # all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)
    
    file_names = [f for f in files_in_subdirs(class_dir, '.ply')]
    random.shuffle(file_names)
    total_num = len(file_names)
    train_files = file_names[:int(total_num*0.9)]
    test_files = file_names[int(total_num*0.9):]
    train_loader = torch.utils.data.DataLoader(ShapenetCore(train_files), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ShapenetCore(test_files), batch_size=args.batch_size, shuffle=True)

    ae = PointNetAutoEncoder(args)
    if torch.cuda.is_available():
        ae.cuda()

    load_pre_trained_ae = True
    restore_epoch = 460
    if load_pre_trained_ae:
        args = Conf.load(train_dir + '/configuration')
        ae = PointNetAutoEncoder(args)
        if torch.cuda.is_available():
            ae.cuda()
        # ae.restore_model(args.train_dir, epoch=restore_epoch)
        ae.load_state_dict(torch.load(osp.join(args.train_dir, "model%d.pth" % restore_epoch)))

    # set optimizer
    optimizer = torch.optim.Adam(ae.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(args.training_epochs):
        # train(train_loader, ae, optimizer, epoch, args)
        test(test_loader, ae, args)

        if epoch % args.saver_step == 0:
            torch.save(ae.state_dict(), osp.join(args.train_dir, "model%d.pth" % epoch))

    """
    buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
    fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
    train_stats = ae.train(all_pc_data, conf, log_file=fout)
    fout.close()
    """