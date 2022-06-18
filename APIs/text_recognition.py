from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class VietOCR:
    def __init__(self, model_path="weights/transformerocr.pth", gpu_enable=False):
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['weights'] = model_path
        config['cnn']['pretrained'] = False
        # config['device'] = 'cuda:0'
        config['device'] = 'cpu' if gpu_enable==False else 'cuda:0'
        config['predictor']['beamsearch'] = False
        self.predictor = Predictor(config)

    def predict(self, img):
        result, prob = self.predictor.predict(img, return_prob=True)

        return result, prob

# out = []
# idx = 0
#     #predict ocr
#     for info, cache in results:
#         ten_sach = ""
#         ten_tac_gia = ""
#         nha_xuat_ban = ""
#         tap = ""
#         nguoi_dich = ""
#         tai_ban = ""
#         for key, value in info.items():
#             for img in value:
#                 if img.shape[0] < img.shape[1] * 2:
#                     s, _ = read(img, key, craft_net, refine_net, detector)
#                     if key == 0:
#                         ten_sach += s + " "
#                     elif key == 1:
#                         ten_tac_gia += s + " "
#                     elif key == 2:
#                         nha_xuat_ban += s + " "
#                     elif key == 3:
#                         tap += s + " "
#                     elif key == 4:
#                         nguoi_dich += s + " "
#                     else:
#                         tai_ban += s + " "
#                 else:
#                     im1 = img
#                     s1, p1 = read(img, key, craft_net, refine_net, detector)

#                     s = s1
#                     p = p1

#                     im2 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#                     s2, p2 = read(im2, key, craft_net, refine_net, detector)

#                     if p2 > p:
#                       p = p2
#                       s = s2

#                     im3 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#                     s3, p3 = read(im3, key, craft_net, refine_net, detector)

#                     if p3 > p:
#                       p = p3
#                       s = s3

#                     if key == 0:
#                         ten_sach += s + " "
#                     elif key == 1:
#                         ten_tac_gia += s + " "
#                     elif key == 2:
#                         nha_xuat_ban += s + " "
#                     elif key == 3:
#                         tap += s + " "
#                     elif key == 4:
#                         nguoi_dich += s + " "
#                     else:
#                         tai_ban += s + " "
#         #nếu mà tên tác giả, nhà xuất bản, tập, người dịch, tái bản có lẫn vào tên sách thì lấy nó ra
#         for i in cache:
#             if i == 1:
#                 if ten_tac_gia in ten_sach:
#                     ten_sach = ten_sach.replace(ten_tac_gia, '')
#                 else:
#                     for s in ten_tac_gia.split():
#                         ten_sach = ten_sach.replace(s, '')
#             elif i == 2:
#                 if nha_xuat_ban in ten_sach:
#                     ten_sach = ten_sach.replace(nha_xuat_ban, '')
#                 else:
#                     for s in nha_xuat_ban.split():
#                         ten_sach = ten_sach.replace(s, '')
#             elif i == 3:
#                 if tap in ten_sach:
#                     ten_sach = ten_sach.replace(tap, '')
#                 else:
#                     for s in tap.split():
#                         ten_sach = ten_sach.replace(s, '')
#             elif i == 4:
#                 if nguoi_dich in ten_sach:
#                     ten_sach = ten_sach.replace(nguoi_dich, '')
#                 else:
#                     for s in nguoi_dich.split():
#                         ten_sach = ten_sach.replace(s, '')
#             elif i == 5:
#                 if tai_ban in ten_sach:
#                     ten_sach = ten_sach.replace(tai_ban, '')
#                 else:
#                     for s in tai_ban.split():
#                         ten_sach = ten_sach.replace(s, '')

#         #thêm vào dictionary
#         features = {
#             'file names' : fn[idx],
#             'tên sách': ten_sach,
#             'tên tác giả': ten_tac_gia,
#             'nhà xuất bản': nha_xuat_ban,
#             'tập': tap,
#             'người dịch': nguoi_dich,
#             'tái bản': tai_ban
#         }

#         idx += 1

    # out.append(features)
