import os.path as osp
import torch.optim

from src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from src.autoencoder import Configuration as Conf
from src.point_net_ae import PointNetAutoEncoder
from src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder
# from src.tf_utils import reset_tf_graph
# from src.general_utils import plot_3d_point_cloud

def train(train_loader, model, optimizer, epoch, args):
    for i, 

    c = configuration
    stats = []

    if c.saver_step is not None:
        create_dir(c.train_dir)

    for _ in range(c.training_epochs):
        # loss, duration = self._single_epoch_train(train_data, c)
        stats.append((epoch, loss, duration))

        if epoch % c.loss_display_step == 0:
            print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
            if log_file is not None:
                log_file.write('%04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

        # Save the models checkpoint periodically.
    return stats

if __name__ == "__main__":
    top_out_dir = './data/'          # Use to save Neural-Net check-points etc.
    top_in_dir = './data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

    experiment_name = 'single_class_ae'
    n_pc_points = 2048                # Number of points per model.
    bneck_size = 128                  # Bottleneck-AE size
    ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'
    # class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
    class_name = "chair"

    syn_id = snc_category_to_synth_id()[class_name]
    class_dir = osp.join(top_in_dir , syn_id)
    all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
    train_dir = create_dir(osp.join(top_out_dir, experiment_name))

    conf = Conf(n_input = [n_pc_points, 3],
                loss = ae_loss,
                train_dir = train_dir,
                encoder = encoder,
                decoder = decoder,
                encoder_args = enc_args,
                decoder_args = dec_args
               )

    conf.experiment_name = experiment_name
    conf.held_out_step = 5   # How often to evaluate/print out loss on 
                             # held_out data (if they are provided in ae.train() ).
    conf.save(osp.join(train_dir, 'configuration'))

    load_pre_trained_ae = False
    restore_epoch = 500
    if load_pre_trained_ae:
        conf = Conf.load(train_dir + '/configuration')
        reset_tf_graph()
        ae = PointNetAutoEncoder(conf.experiment_name, conf)
        ae.restore_model(conf.train_dir, epoch=restore_epoch)

    ae = PointNetAutoEncoder(conf.experiment_name, conf)

    # set optimizer
    optimizer = torch.optim.Adam(ae.parameters(), conf.learning_rate, weight_decay=conf.weight_decay)

    for epoch in 

    """
    buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
    fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
    train_stats = ae.train(all_pc_data, conf, log_file=fout)
    fout.close()
    """