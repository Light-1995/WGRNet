import torch
import scipy.io as sio
from utils.ssim import *
import numpy
from model import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from data import *
from utils.metrics import *
from PIL import Image
import cv2

if __name__ == '__main__':
    spectral_num = 4
    image_scale = 1023
    ergas = 0
    sam = 0
    cc = 0
    psnr = 0
    rmse = 0
    ssim = 0
    num_test = 20 

    test_set = Dataset_Pro('../dataset/test_gf2_multiExm1.h5')
    test_dataloader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False,
                                 pin_memory=True,
                                 drop_last=True)
    
    model = ADKNet(spectral_num=spectral_num).cuda()
    model.load_state_dict(torch.load('./Weights/best_gf2.pth'))
    
    with torch.no_grad():
        for iteration, data in enumerate(test_dataloader, 0):
            gt, lms, ms, pan = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda()
            HRMS = model(ms, pan)
            
            ergas += ERGAS(HRMS, gt, 2)
            sam += SAM(HRMS, gt)
            cc += cross_correlation(HRMS, gt)
            psnr += PSNR(HRMS, gt)
            rmse += RMSE(HRMS, gt)
            ssim += Ssim(HRMS, gt)

            
            GT = torch.squeeze(gt).cpu().detach().numpy()
            image1 = np.moveaxis(GT, 0, -1)
            print(image1.shape)
            filename1 = f"./GT/gf2/gt/GT{iteration + 1}.mat"
            sio.savemat(filename1, {'data': image1})
            
            MS = torch.squeeze(HRMS).cpu().detach().numpy()
            image2 = np.moveaxis(MS, 0, -1)
            filename2 = f"./HRMS/gf2/hrms/MS{iteration + 1}.mat"
            sio.savemat(filename2, {'data': image2})

            HRMS = HRMS.cpu().numpy()

            
            imagem1 = HRMS[0, 2, :, :]
            imagem2 = HRMS[0, 1, :, :]
            imagem3 = HRMS[0, 0, :, :]
            ms = np.stack((imagem1, imagem2, imagem3), axis=2)
            saturation_factor = 1.5
            
            image_hsv = cv2.cvtColor(ms, cv2.COLOR_BGR2HSV)
            image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * saturation_factor, 0.01, 0.99)
            ms = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR) * image_scale
            
            gt = gt.cpu().numpy()
            imageg1 = gt[0, 2, :, :]
            imageg2 = gt[0, 1, :, :]
            imageg3 = gt[0, 0, :, :]
            gt = np.stack((imageg1, imageg2, imageg3), axis=2)
            image_hsv = cv2.cvtColor(gt, cv2.COLOR_BGR2HSV)
            image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1] * saturation_factor, 0.01, 0.99)
            gt = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR) * image_scale
            
            min_target = 0
            max_target = 255
            # 执行线性归一化
            ms = ((ms - ms.min()) / (ms.max() - ms.min())) * (max_target - min_target) + min_target
            print(np.min(ms))
            print(np.max(ms))
            ms = ms.astype(numpy.uint8)
            gt = ((gt - gt.min()) / (gt.max() - gt.min())) * (max_target - min_target) + min_target
            gt = gt.astype(numpy.uint8)

            ms = Image.fromarray(np.uint8(ms), 'RGB')
            gt = Image.fromarray(np.uint8(gt), 'RGB')
            
            
            filename3 = f"./HRMS/gf2/hrms_RGB/ms{iteration+1}.tif"
            filename4 = f"./GT/gf2/gt_RGB/gt{iteration + 1}.tif"
            ms.save(filename3)
            gt.save(filename4)
            
        
        ergas = ergas / num_test
        sam = sam / num_test
        cc = cc / num_test
        psnr = psnr / num_test
        rmse = rmse / num_test
        ssim = ssim / num_test
        
        print("  ergas:{:.4f}   cc:{:.4f} sam:{:.4f}  psnr:{:.4f} rmse:{:.4f} ssim:{:.4f}".format(
            ergas, cc, sam, psnr, rmse, ssim))




