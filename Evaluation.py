import torch

from network.transEM.build import build_mutitask_dn
from build_data import *
from module_list import *

from util.generate_heatmap import *
from util.calculate_jac_dice_dq_ import *
from util.sliding_inferece import inference_det

model = build_mutitask_dn(image_size = 384)
model.load_state_dict(torch.load('H.pth'))
# model = nn.DataParallel(model)
device = torch.device("cuda:0")
model = model.to(device)

total_aji = 0.0
total_pq = 0.0
total_dice = 0.0
total_jac = 0.0
num = 0
jac_list = []
adress = glob.glob('/home/icml/code/trans_adap/dataset/H/test/img/im*.png')

adress.sort()
idx_list = [int(file[re.search(r'im\d',file).span()[1]: file.rfind('.')]) for file in adress]
for i in idx_list:
    im_id = str(i)
    im = Image.open('/home/icml/code/trans_adap/dataset/H/test/img/im{}.png'.format(im_id.zfill(4))) 
    label = np.array(Image.open('/home/icml/code/trans_adap/dataset/H/test/lab/im{}.png'.format(im_id.zfill(4))))
    point = Image.open('/home/icml/code/trans_adap/dataset/H/test/point/im{}.png'.format(im_id.zfill(4)))
    im_size = [label.shape[0], label.shape[1]]
    label[label == 255] = 1
    label[label == 128] = 0
    label = Image.fromarray(label)
    img = Image.fromarray(np.stack((im,im,im),axis = -1))
    im_tensor, label_tensor,heatmap= transform(img, label, None, point = point, crop_size=im_size, scale_size=(1.0, 1.0),augmentation=False)
    im_tensor = im_tensor.to('cuda:0')
    with torch.no_grad():
        heatmap = torch.zeros_like(heatmap)
        pred,_ = inference_det(im_tensor.unsqueeze(0),model, heatmap.unsqueeze(0),
                               384,128,50, is_aug=True)
        pre=(pred.argmax(1)[0].numpy()).astype(np.uint8)
        true=(label_tensor[0].cpu().numpy()).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(pre)
        for t in range(num_labels):
            if(np.sum(labels == t) <=15 ):
                pre[labels == t] = 0
        aji = get_fast_aji(true, pre)
        pq = get_fast_pq(true, pre)
        dice = 2*np.sum(true*pre)/(np.sum(true)+np.sum(pre))
        jac = np.sum(true*pre)/np.sum(np.logical_or(true,pre))
        jac_list.append(jac)
        print( "{}.png  aji:{:.3f},pq:{:.3f},dice:{:.3f},jac:{:.3f}".format(i,aji,pq[2],dice,jac))
        total_aji += aji
        total_pq += pq[2]
        total_dice += dice
        total_jac += jac
        num +=1
print("total_aji:{:.5f},total_pq:{:.5f},total_dice:{:.5f},total_jac:{:.5f}".format(
    total_aji/num, total_pq/num, total_dice/num, total_jac/num) )