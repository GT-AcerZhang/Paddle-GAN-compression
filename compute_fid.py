from metric.compute_fid import calculate_fid_given_paths

paths = ['cycle_gan/data/horse2zebra/testB/', '/root/work/GAN-compression/Paddle-GAN-compression/cycle_gan_mobile/output_138/eval/61/fakeB/']
#paths = ['cycle_gan/data/horse2zebra/testB/', 'cycle_gan/output_0/eval/183/fakeB/']
#paths = ['cycle_gan/data/horse2zebra/testB/', '/root/work/GAN-compression/pytorch-CycleGAN-and-pix2pix/results/horse2zebra_pretrained/test_latest/images/fake/']
fid_value = calculate_fid_given_paths(paths=paths, batch_size=1, use_gpu=True, dims=2048, premodel_path='./metric/params_inceptionV3')

print("FID: ", fid_value)
