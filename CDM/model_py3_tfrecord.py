import time

from utils_py3_tfrecord import *
from model_database import *
import rawpy

class denoiser(object):
    def __init__(self, sess, input_c_dim=3, batch_size=64, patch_size=100):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='GroundTruth') # ground truth
        self.X = tf.placeholder(tf.float32, [None, None, None, 1], name='BayerRawData') # input of the network
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.Y = CIR_CDM(self.X, is_training=self.is_training)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def test(self, test_files_gt, ckpt_dir, save_dir):
        # init variables
        tf.global_variables_initializer().run()
        assert len(test_files_gt) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")
        psnr_sum = 0
        test_sum = 0
        msssim_sum = 0
        ssim_sum = 0
        csnr_sum = np.zeros(3)
        for idx in range(len(test_files_gt)):
            imagename = os.path.basename(test_files_gt[idx])
            clean_image = load_images(test_files_gt[idx]).astype(np.float32)
            _, w, h, _  = clean_image.shape
            clean_image = clean_image[:, 0:w//2*2, 0:h//2*2, :]
            image_bayer = getBayer(load_images(test_files_gt[idx])[:, 0:w//2*2, 0:h//2*2, :]).astype(np.float32)
            image_bayer = np.expand_dims(image_bayer, 0)
            image_bayer = np.expand_dims(image_bayer, -1)
            test_s_time = time.time()
            output_clean_image = self.sess.run(self.Y, feed_dict={self.X: image_bayer, self.is_training: False})
            test_time = time.time()-test_s_time
            groundtruth = np.clip(clean_image, 0, 255).astype('uint8')
            outputimage = np.around(np.clip(output_clean_image, 0, 255)).astype('uint8')
            psnr = imcpsnr(groundtruth, outputimage, 255, 10)
            csnr = impsnr(groundtruth, outputimage, 255, 10)
            msssim = MS_SSIM(groundtruth, outputimage, 10)
            ssim = self.sess.run(SSIM(groundtruth, outputimage, 10)) 
            print("%s, Final PSNR: %.2fdB (R: %.2f, G: %.2f, B: %.2f), MSSSIM: %.5f, Time: %.4fs" % (imagename, psnr, csnr[0], csnr[1], csnr[2], msssim, test_time))
            psnr_sum += psnr
            csnr_sum += csnr
            msssim_sum += msssim
            ssim_sum += ssim
            test_sum += test_time
            save_images(os.path.join(save_dir, imagename), outputimage)
        avg_psnr = psnr_sum / len(test_files_gt)
        avg_csnr = csnr_sum / len(test_files_gt)
        avg_msssim = msssim_sum / len(test_files_gt)
        avg_ssim = ssim_sum / len(test_files_gt)
        print("--- Test --- Average PSNR Final: %.2fdB (R: %.2f, G: %.2f, B: %.2f), MSSSIM: %.5f, SSIM: %.5f, Running Time: %.4fs ---" % (avg_psnr, avg_csnr[0], avg_csnr[1], avg_csnr[2], avg_msssim, avg_ssim, test_sum))

    def self_ensemble_test(self, test_files_gt, ckpt_dir, save_dir):
        # init variables
        tf.global_variables_initializer().run()
        assert len(test_files_gt) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")
        psnr_sum = 0
        msssim_sum = 0
        ssim_sum = 0
        csnr_sum = np.zeros(3)
        for idx in range(len(test_files_gt)):
            imagename = os.path.basename(test_files_gt[idx])
            clean_image = load_images(test_files_gt[idx]).astype(np.float32)
            _, w, h, _  = clean_image.shape
            clean_image = clean_image[:, 0:w//2*2, 0:h//2*2, :]
            image_bayer = getBayer(load_images(test_files_gt[idx])[:, 0:w//2*2, 0:h//2*2, :]).astype(np.float32)
            image_bayer = np.expand_dims(image_bayer, 0)
            image_bayer = np.expand_dims(image_bayer, -1)
            image_ensemble = np.zeros([8,image_bayer.shape[1],image_bayer.shape[2],3])
            # mode 1-8
            for mode in range(8):
                if mode == 0:
                    image_bayerRGGB = image_bayer
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.X: image_bayerRGGB, self.is_training: False})
                    image_ensemble[mode,:,:,:] = output_clean_image
                    print(imcpsnr(clean_image, output_clean_image, 255, 10))
                elif mode == 1:
                    image_bayer1 = np.flip(image_bayer,2)
                    image_bayerRGGB = image_bayer1[:,:,1:image_bayer1.shape[2]-1,:]
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.X: image_bayerRGGB, self.is_training: False})
                    output_clean_image1 = np.pad(output_clean_image, pad_width=((0,0),(0,0),(1,1),(0,0)), mode='symmetric')
                    image_ensemble[mode,:,:,:] = np.flip(output_clean_image1,2)
                    print(imcpsnr(clean_image, np.flip(output_clean_image1,2), 255, 10))
                elif mode == 2:
                    image_bayer1 = np.flip(image_bayer,1)
                    image_bayerRGGB = image_bayer1[:,1:image_bayer1.shape[1]-1,:,:]
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.X: image_bayerRGGB, self.is_training: False})
                    output_clean_image1 = np.pad(output_clean_image, pad_width=((0,0),(1,1),(0,0),(0,0)), mode='symmetric')
                    image_ensemble[mode,:,:,:] = np.flip(output_clean_image1,1)
                    print(imcpsnr(clean_image, np.flip(output_clean_image1,1), 255, 10))
                elif mode == 3:
                    image_bayer1 = np.rot90(image_bayer,axes=(1,2))
                    image_bayerRGGB = image_bayer1[:,1:image_bayer1.shape[1]-1,:,:]
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.X: image_bayerRGGB, self.is_training: False})
                    output_clean_image1 = np.pad(output_clean_image, pad_width=((0,0),(1,1),(0,0),(0,0)), mode='symmetric')
                    image_ensemble[mode,:,:,:] = np.rot90(output_clean_image1,3,axes=(1,2))
                    print(imcpsnr(clean_image, np.rot90(output_clean_image1,3,axes=(1,2)), 255, 10))
                elif mode == 4:
                    image_bayer1 = np.rot90(image_bayer,2,axes=(1,2))
                    image_bayerRGGB = image_bayer1[:,1:image_bayer1.shape[1]-1,1:image_bayer1.shape[2]-1,:]
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.X: image_bayerRGGB, self.is_training: False})
                    output_clean_image1 = np.pad(output_clean_image, pad_width=((0,0),(1,1),(1,1),(0,0)), mode='symmetric')
                    image_ensemble[mode,:,:,:] = np.rot90(output_clean_image1,2,axes=(1,2))
                    print(imcpsnr(clean_image, np.rot90(output_clean_image1,2,axes=(1,2)), 255, 10))
                elif mode == 5:
                    image_bayer1 = np.rot90(image_bayer,3,axes=(1,2))
                    image_bayerRGGB = image_bayer1[:,:,1:image_bayer1.shape[2]-1,:]
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.X: image_bayerRGGB, self.is_training: False})
                    output_clean_image1 = np.pad(output_clean_image, pad_width=((0,0),(0,0),(1,1),(0,0)), mode='symmetric')
                    image_ensemble[mode,:,:,:] = np.rot90(output_clean_image1,axes=(1,2))
                    print(imcpsnr(clean_image, np.rot90(output_clean_image1,axes=(1,2)), 255, 10))
                elif mode == 6:
                    image_bayer1 = np.flip(np.rot90(image_bayer,axes=(1,2)),2)
                    image_bayerRGGB = image_bayer1[:,1:image_bayer1.shape[1]-1,1:image_bayer1.shape[2]-1,:]
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.X: image_bayerRGGB, self.is_training: False})
                    output_clean_image1 = np.pad(output_clean_image, pad_width=((0,0),(1,1),(1,1),(0,0)), mode='symmetric')
                    image_ensemble[mode,:,:,:] = np.rot90(np.flip(output_clean_image1,2),3,axes=(1,2))
                    print(imcpsnr(clean_image, np.rot90(np.flip(output_clean_image1,2),3,axes=(1,2)), 255, 10))
                elif mode == 7:
                    image_bayer1 = np.flip(np.rot90(image_bayer,3,axes=(1,2)),2)
                    image_bayerRGGB = image_bayer1
                    output_clean_image = self.sess.run(self.Y, feed_dict={self.X: image_bayerRGGB, self.is_training: False})
                    image_ensemble[mode,:,:,:] = np.rot90(np.flip(output_clean_image,2),axes=(1,2))
                    print(imcpsnr(clean_image, np.rot90(np.flip(output_clean_image,2),axes=(1,2)), 255, 10))
                else:
                    print('[!]Wrong Mode')
                    exit(0)
            groundtruth = np.clip(clean_image, 0, 255).astype('uint8')
            outputimage = np.average(image_ensemble,axis=0)
            outputimage = np.around(np.clip(outputimage, 0, 255)).astype('uint8')
            outputimage = np.expand_dims(outputimage, 0)
            psnr = imcpsnr(groundtruth, outputimage, 255, 10)
            csnr = impsnr(groundtruth, outputimage, 255, 10)
            msssim = MS_SSIM(groundtruth, outputimage, 10)
            ssim = self.sess.run(SSIM(groundtruth, outputimage, 10)) 
            print("%s, Final PSNR: %.2fdB (R: %.2f, G: %.2f, B: %.2f), MSSSIM: %.5f" % (imagename, psnr, csnr[0], csnr[1], csnr[2], msssim))
            psnr_sum += psnr
            csnr_sum += csnr
            msssim_sum += msssim
            ssim_sum += ssim
            save_images(os.path.join(save_dir, imagename), outputimage)
        avg_psnr = psnr_sum / len(test_files_gt)
        avg_csnr = csnr_sum / len(test_files_gt)
        avg_msssim = msssim_sum / len(test_files_gt)
        avg_ssim = ssim_sum / len(test_files_gt)
        print("--- Test --- Average PSNR Final: %.2fdB (R: %.2f, G: %.2f, B: %.2f), MSSSIM: %.5f, SSIM: %.5f ---" % (avg_psnr, avg_csnr[0], avg_csnr[1], avg_csnr[2], avg_msssim, avg_ssim))

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0
