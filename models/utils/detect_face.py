import torch

from torch.nn.modules.module import Module
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from torchvision.ops.boxes import batched_nms
from PIL import Image
import numpy as np
import os
import math
import typing

from ..modules import PNet, RNet, ONet
from .tracing import nvtx_range

# OpenCV is optional, but required if using numpy arrays instead of PIL
try:
    import cv2
except:
    pass

def fixed_batch_process(im_data, model):
    batch_size = 512
    out = []
    for i in range(0, len(im_data), batch_size):
        batch = im_data[i:(i+batch_size)]
        out.append(model(batch))

    return tuple(torch.cat(v, dim=0) for v in zip(*out))

def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device):
    '''Remove control flow - we can just assert that imgs is a pytorch.Tensor
        The function is already written to be able to operate on torch tensors
        that are on the GPU.
    '''
    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        if isinstance(imgs,np.ndarray):
            imgs = torch.as_tensor(imgs.copy(), device=device)

        if isinstance(imgs,torch.Tensor):
            imgs = torch.as_tensor(imgs, device=device)

        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        if any(img.size != imgs[0].size for img in imgs):
            raise Exception("MTCNN batch processing only compatible with equal-dimension images.")
        imgs = np.stack([np.uint8(img) for img in imgs])
        imgs = torch.as_tensor(imgs.copy(), device=device)

    

    model_dtype = next(pnet.parameters()).dtype
    imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m

    # Create scale pyramid
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor

    # First stage
    boxes = []
    image_inds = []

    scale_picks = []

    all_i = 0
    offset = 0
    for scale in scales:
        im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)
    
        boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, threshold[0])
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)

        pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5)
        scale_picks.append(pick + offset)
        offset += boxes_scale.shape[0]

    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0)

    scale_picks = torch.cat(scale_picks, dim=0)

    # NMS within each scale + image
    boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]


    # NMS within each image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = rerec(boxes)
    y, ey, x, ex = pad(boxes, w, h)
    
    
    # Second stage
    if len(boxes) > 0:
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (24, 24)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125

        # This is equivalent to out = rnet(im_data) to avoid GPU out of memory.
        out = fixed_batch_process(im_data, rnet)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        score = out1[1, :]
        ipass = score > threshold[1]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
        boxes = bbreg(boxes, mv)
        boxes = rerec(boxes)

    # Third stage
    points = torch.zeros(0, 5, 2, device=device)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h)
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (48, 48)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125
        
        # This is equivalent to out = onet(im_data) to avoid GPU out of memory.
        # This can possibly be reverted to just an onet call and we can control
        # oom by limiting max batch size of the model at the config level
        out = fixed_batch_process(im_data, onet)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)
        score = out2[1, :]
        points = out1
        ipass = score > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)
        boxes = bbreg(boxes, mv)

        # NMS within each image using "Min" strategy
        # pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    #Remove conversion back to numpy and just return tensors
    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()

    image_inds = image_inds.cpu()

    batch_boxes = []
    batch_points = []
    for b_i in range(batch_size):
        b_i_inds = np.where(image_inds == b_i)
        batch_boxes.append(boxes[b_i_inds].copy())
        batch_points.append(points[b_i_inds].copy())

    batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)

    return batch_boxes, batch_points

def detect_face_scripted(imgs: torch.Tensor, minsize: int, pnet: PNet, rnet: RNet, onet: ONet, threshold: typing.List[float], factor: float, device: torch.device):
    '''Remove control flow - we can just assert that imgs is a pytorch.Tensor
        The function is already written to be able to operate on torch tensors
        that are on the GPU.
    '''  
    with nvtx_range('Preprocess'):
        model_dtype = torch.float32
        imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

        batch_size = len(imgs)
        h, w = imgs.shape[2:4]
        m = 12.0 / minsize
        minl = min(h, w)
        minl = minl * m

        # Create scale pyramid
        start = 0
        end = math.floor(math.log(12 / minl) / math.log(factor))
        scales = torch.logspace(end, start, end, factor).multiply(m)

        # First stage
        boxes = []
        image_inds = []

        scale_picks = []

        offset = 0

    with nvtx_range('pnet'):
        with nvtx_range('pnet:scales'):
            for scale in scales:
                with nvtx_range('pnet:scales:interpolate'):
                    im_data = interpolate(imgs, (int(h * scale + 1), int(w * scale + 1)), mode='area')
                im_data = (im_data - 127.5) * 0.0078125

                with nvtx_range('pnet:scales:forward'):
                    reg, probs = pnet.forward(im_data)
                    probs.to(torch.device('cpu'))
                with nvtx_range('pnet:scales:generate_bounding_box'): 
                    boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, threshold[0])
                boxes.append(boxes_scale)
                image_inds.append(image_inds_scale)

                with nvtx_range('pnet:scales:nms'):
                    pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5)
                scale_picks.append(pick + offset)
                offset += boxes_scale.shape[0]

        with nvtx_range('pnet:nms'):
            boxes = torch.cat(boxes, dim=0)
            image_inds = torch.cat(image_inds, dim=0)

            scale_picks = torch.cat(scale_picks, dim=0)

            # NMS within each scale + image
            boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]


            # NMS within each image
            pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        
        with nvtx_range('pnet:postprocess'):
            boxes, image_inds = boxes[pick], image_inds[pick]

            regw = boxes[:, 2] - boxes[:, 0]
            regh = boxes[:, 3] - boxes[:, 1]
            qq1 = boxes[:, 0] + boxes[:, 5] * regw
            qq2 = boxes[:, 1] + boxes[:, 6] * regh
            qq3 = boxes[:, 2] + boxes[:, 7] * regw
            qq4 = boxes[:, 3] + boxes[:, 8] * regh
            boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
            boxes = rerec(boxes)
            y, ey, x, ex = pad(boxes, w, h)
            y = y.cpu()
            ey = ey.cpu()
            x = x.cpu()
            ex = ex.cpu()
            image_inds_cpu = image_inds.clone().to(torch.device('cpu'))
    
    with nvtx_range('rnet'):
        # Second stage
        if len(boxes) > 0:
            with nvtx_range('rnet:resample'):
                im_data = []
                for k in range(len(y)):
                    img_k = imgs[image_inds_cpu[k], :, y[k]:ey[k], x[k]:ex[k]].unsqueeze(0)
                    img_k = interpolate(img_k, (24, 24), mode='area')
                    im_data.append(img_k)
            
                im_data = torch.cat(im_data, dim=0)
                im_data = (im_data - 127.5) * 0.0078125

            with nvtx_range('rnet:forward'):
            # This is equivalent to out = rnet(im_data) to avoid GPU out of memory.
                out = rnet.forward(im_data)

            with nvtx_range('rnet:nms'):
                out0 = out[0].permute(1, 0)
                out1 = out[1].permute(1, 0)
                score = out1[1, :]
                ipass = score > threshold[1]
                boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
                image_inds = image_inds[ipass]
                mv = out0[:, ipass].permute(1, 0)

                # NMS within each image
                pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
                boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
            boxes = bbreg(boxes, mv)
            boxes = rerec(boxes)

    with nvtx_range('onet'):
        # Third stage
        points = torch.zeros(0, 5, 2, device=device)
        if len(boxes) > 0:
            with nvtx_range('onet:resample'):
                y, ey, x, ex = pad(boxes, w, h)
                y = y.cpu()
                ey = ey.cpu()
                x = x.cpu()
                ex = ex.cpu()
                image_inds_cpu = image_inds.clone().to(torch.device('cpu'))
                im_data = []
                for k in range(len(y)):
                    img_k = imgs[image_inds_cpu[k], :, y[k]:ey[k], x[k]:ex[k]].unsqueeze(0)
                    img_k = interpolate(im_data, (48, 48), mode='area')
                    im_data.append(img_k)
                im_data = torch.cat(im_data, dim=0)
                im_data = (im_data - 127.5) * 0.0078125
            
            # This is equivalent to out = onet(im_data) to avoid GPU out of memory.
            # This can possibly be reverted to just an onet call and we can control
            # oom by limiting max batch size of the model at the config level
            with nvtx_range('onet:forward'):
                out = onet.forward(im_data)
            
            with nvtx_range('onet:nms'):
                out0 = out[0].permute(1, 0)
                out1 = out[1].permute(1, 0)
                out2 = out[2].permute(1, 0)
                score = out2[1, :]
                points = out1
                ipass = score > threshold[2]
                points = points[:, ipass]
                boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
                image_inds = image_inds[ipass]
                mv = out0[:, ipass].permute(1, 0)

                w_i = boxes[:, 2] - boxes[:, 0] + 1
                h_i = boxes[:, 3] - boxes[:, 1] + 1
                points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
                points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
                points = torch.stack((points_x, points_y)).permute(2, 1, 0)
                boxes = bbreg(boxes, mv)

                # NMS within each image using "Min" strategy
                # pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
                pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
            boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    with nvtx_range('post modeling'):
        #Remove conversion back to numpy and just return tensors
        boxes = boxes.cpu()
        points = points.cpu()

        image_inds = image_inds.cpu()

        batch_boxes = []
        batch_points = []
        for b_i in range(batch_size):
            b_i_inds = torch.where(image_inds == b_i)
            batch_boxes.append(boxes.index_select(0, b_i_inds[0]))
            batch_points.append(points.index_select(0, b_i_inds[0]))

        batch_boxes, batch_points = torch.cat(batch_boxes), torch.cat(batch_points)

    return batch_boxes, batch_points


def bbreg(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return boundingbox

def generateBoundingBoxesTopK(reg: torch.Tensor, probs: torch.Tensor, scale: float, thresh: float):
    stride = 2
    cellsize = 12
    reg = reg.permute(1, 0, 2, 3)
    sorted_ind = probs.argsort(descending=True)
    probs_sorted = probs.clone().sort(descending=True)
    probs_sorted
    return None

def generateBoundingBox(reg: torch.Tensor, probs: torch.Tensor, scale: float, thresh: float):
    '''
    reg: (N, 4, H, W)
    probs: (N, H, W)
    '''
    stride = 2 # Convolution stride length of output kernel of pnet
    cellsize = 12 # Matches m Scaling Factor from line 211

    reg = reg.permute(1, 0, 2, 3)
    # (4, N, H, W)

    mask = (probs >= thresh)
    #(N, H, W)
    with nvtx_range('generate_bounding_box:mask_nonzero'):
        mask_inds = mask.nonzero()
        #(0 <= I <= H*W, 3) indices of nonzero elements in 3D tensor
    with nvtx_range('generate_bounding_box:mask_indexing'):
        image_inds = mask_inds[:, 0] #zeros when N = 1. Indicates which image
        score = probs[mask].pin_memory()
        score.to(reg.get_device(), non_blocking=True)
        #(I)
        reg = reg[:, mask].permute(1, 0)
        #(4, I) -> (I, 4)
        bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
        #(I, 2) Elements are indices from probs that are nonzero after thresholding flipped over axis 1 (W,H)
    q1 = ((stride * bb + 1) / scale).floor().pin_memory()
    q1.to(reg.get_device(), non_blocking=True)
    # Relative X positions on Image of detection
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor().pin_memory()
    q2.to(reg.get_device(), non_blocking=True)
    # Relative Y positions on Image of detection
    torch.cuda.synchronize()
    with nvtx_range('generate_bounding_box:tensor_creation'):
        boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def nms_numpy(boxes, scores, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]

        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()

        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = w * h
        if method == 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]

    pick = pick[:counter].copy()
    return pick


def batched_nms_numpy(boxes, scores, idxs, threshold, method):
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def pad(boxes: torch.Tensor, w: int, h: int):
    x = boxes[:, 0].int()
    y = boxes[:, 1].int()
    ex = boxes[:, 2].int()
    ey = boxes[:, 3].int()

    maxw_mask = ex > w
    maxh_mask = ey > h
    minw_mask = x < 0
    minh_mask = y < 0

    #expand boxes where the standard size crop will be oob and clamped.
    x[maxw_mask] = x[maxw_mask] - (ex[maxw_mask] - w)
    y[maxh_mask] = y[maxh_mask] - (ey[maxh_mask] - h)
    ex[minw_mask] = ex[minw_mask] - x[minw_mask]
    ey[minh_mask] = ey[minh_mask] - y[minh_mask]


    x[minw_mask] = 0
    y[minh_mask] = 0
    ex[maxw_mask] = w
    ey[maxh_mask] = h
    
    return y, ey, x, ex


def rerec(bboxA):
    if bboxA.shape[0] == 0:
        return bboxA
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    
    l = torch.max(w, h)
    bboxA[:, 0] = bboxA[:, 0] + torch.floor(w * 0.5) - torch.floor(l * 0.5)
    bboxA[:, 1] = bboxA[:, 1] + torch.floor(h * 0.5) - torch.floor(l * 0.5)
    bboxA[:, 2:4] = (bboxA[:, :2] + l).floor()

    return bboxA


def imresample(img: torch.Tensor, sz: typing.Tuple[int, int]) -> torch.Tensor:
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out


def save_img(img, path):
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.
    
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})
    
    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    face = F.to_tensor(np.float32(face))

    return face
