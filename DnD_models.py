import sys
import os
sys.path.append(".")
from openai import OpenAI
import data_utils
import utils
from PIL import ImageDraw
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
from torchvision import transforms
from io import BytesIO
import base64
import cv2
import matplotlib.pyplot as plt
import functools
import json
import requests




def get_attention_crops(target_name, images, neuron_id, num_crops_per_image = 4, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg', return_bounding_box = False, model=None, preprocess=None):
    
    if target_name == 'custom':
        target_model, preprocess = data_utils.get_target_model(target_name, device, model, preprocess)
    else:
        target_model, preprocess = data_utils.get_target_model(target_name, device)
    all_features = {target_layer:[] for target_layer in target_layers}
    hooks = {}
    
    transform = transforms.ToPILImage()
    
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(utils.get_mean_activation(all_features[target_layer]))".format(target_layer)
        hooks[target_layer] = eval(command)
        
    with torch.no_grad():
        for image in images:
            if "tile2vec" not in target_name:
                features = target_model(preprocess(image).unsqueeze(0).to(device))
            elif "custom" in target_name:
                features = target_model(preprocess(np.array(image)).unsqueeze(0).to(device))
            else:
                features = target_model.encode(image.unsqueeze(0).to(device))
    
    all_heatmaps = {target_layer:[] for target_layer in target_layers}
    for target_layer in target_layers:
        all_features[target_layer] = torch.cat(all_features[target_layer])
        hooks[target_layer].remove()
        
        for i in range(len(all_features[target_layer])):
            if "tile2vec" in target_name:
                images[i] = Image.fromarray(np.uint8(images[i].permute(1,2,0).numpy() * 255))
                
            if target_layer != 'fc' and target_layer != 'encoder':
                heatmap = transform(all_features[target_layer][i][neuron_id])
            elif target_layer == 'encoder':
                unflattend_img = torch.unflatten(all_features[target_layer][i],0,(16,16,3))
                unflattend_img = torch.permute(unflattend_img, (2,0,1))
                heatmap = ImageOps.grayscale(transform(unflattend_img))
            else:
                heatmap = transform(all_features[target_layer][i])
            heatmap = heatmap.resize([images[i].size[0],images[i].size[1]])
            heatmap = np.array(heatmap)
            all_heatmaps[target_layer].append(heatmap)
        if(return_bounding_box == True):
            utils.show_binarized_heatmap(all_heatmaps[target_layer])
    
    all_image_crops = [];
    all_bb_box = {layer : {i:[] for i in range(len(all_heatmaps[target_layer]))} for layer in target_layers}
    thresholded_feature_maps = []
    thresholds = []
    for target_layer in target_layers:
        for i, heatmap in enumerate(all_heatmaps[target_layer]): 
            thresh_val, thresh = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            thresholded_feature_maps.append(thresh)
            thresholds.append(thresh_val)
           
            bb_cor = []
            # Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                box = (x, y, x + w, y + h)
                bb_cor.append(box)
                
            bb_cor = sorted(bb_cor, key=functools.cmp_to_key(utils.compare))
            
            cropped_bb = []
            for box in bb_cor:
                if len(cropped_bb) == num_crops_per_image:
                    break
                p = 0
                good_to_add = True
                while p < len(cropped_bb):
                    if utils.IoU(box, cropped_bb[p]) <= 0.5: 
                        p += 1
                    else:
                        good_to_add = False
                        break
                if good_to_add and utils.IoU(box,(0,0,heatmap.shape[0],heatmap.shape[1])) < 0.8:
                    cropped_img = images[i].crop(box)
                    cropped_img = cropped_img.resize([heatmap.shape[0],heatmap.shape[1]])
                    if "tile2vec" in target_name:
                        cropped_img = (torch.from_numpy(np.float64(np.array(cropped_img)) / 255).permute(2, 0, 1)).type(torch.FloatTensor)
                    all_image_crops.append(cropped_img)
                    cropped_bb.append(box)
            all_bb_box[target_layer][i] = cropped_bb
        if(return_bounding_box == True):
            utils.show_otsu_threshold(thresholded_feature_maps, thresholds)
            utils.show_bbox_on_heatmap(all_heatmaps[target_layer], all_bb_box[target_layer])  
    if return_bounding_box == True:
        return all_bb_box[target_layers[0]], all_image_crops
    else:
        del all_bb_box
        return all_image_crops





def get_attention_crops_brain(target_name, images, neuron_id, num_crops_per_image = 4, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg', return_bounding_box = False, model=None, preprocess=None,weights=None):
    
    if target_name == 'custom':
        target_model, preprocess = data_utils.get_target_model(target_name, device, model, preprocess)
    else:
        target_model, preprocess = data_utils.get_target_model(target_name, device)
    all_features = {target_layer:[] for target_layer in target_layers}
    hooks = {}
    
    transform = transforms.ToPILImage()
    
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(utils.get_mean_activation_brain(all_features[target_layer],weights))".format(target_layer)
        hooks[target_layer] = eval(command)
        
    with torch.no_grad():
        for image in images:
            if "tile2vec" not in target_name:
         
                #if target_name=="clip_resnet50":
                if "clip" in target_name:  
                    features = target_model.encode_image(preprocess(image).unsqueeze(0).to(device))  # 
                else:
                    features = target_model(preprocess(image).unsqueeze(0).to(device))
           
            elif "custom" in target_name:
                features = target_model(preprocess(np.array(image)).unsqueeze(0).to(device))
            else:
                features = target_model.encode(image.unsqueeze(0).to(device))
    
    all_heatmaps = {target_layer:[] for target_layer in target_layers}

    for target_layer in target_layers:
        all_features[target_layer] = torch.cat(all_features[target_layer])
        hooks[target_layer].remove()
        

        
        for i in range(len(all_features[target_layer])):
            #print(all_features[target_layer][i].shape)
            if "tile2vec" in target_name:
                images[i] = Image.fromarray(np.uint8(images[i].permute(1,2,0).numpy() * 255))
            if target_layer != 'fc' and target_layer != 'encoder':
                heatmap = transform(all_features[target_layer][i][neuron_id])          
            #
            elif target_layer == 'encoder':
                unflattend_img = torch.unflatten(all_features[target_layer][i],0,(16,16,3))
                unflattend_img = torch.permute(unflattend_img, (2,0,1))
                heatmap = ImageOps.grayscale(transform(unflattend_img))
            else:
                heatmap = transform(all_features[target_layer][i])
            heatmap = heatmap.resize([images[i].size[0],images[i].size[1]])
            heatmap = np.array(heatmap)
            all_heatmaps[target_layer].append(heatmap)
        if(return_bounding_box == True):
            utils.show_binarized_heatmap(all_heatmaps[target_layer])
    
    all_image_crops = [];
    all_bb_box = {layer : {i:[] for i in range(len(all_heatmaps[target_layer]))} for layer in target_layers}
    thresholded_feature_maps = []
    thresholds = []
    for target_layer in target_layers:
        for i, heatmap in enumerate(all_heatmaps[target_layer]): 
            thresh_val, thresh = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresholded_feature_maps.append(thresh)
            thresholds.append(thresh_val)
            bb_cor = []
            # Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                box = (x, y, x + w, y + h)
                bb_cor.append(box)
            bb_cor = sorted(bb_cor, key=functools.cmp_to_key(utils.compare))
            cropped_bb = []
            for box in bb_cor:
                if len(cropped_bb) == num_crops_per_image:
                    break
                p = 0
                good_to_add = True
                while p < len(cropped_bb):
                    if utils.IoU(box, cropped_bb[p]) <= 0.5: 
                        p += 1
                    else:
                        good_to_add = False
                        break

                if good_to_add and utils.IoU(box,(0,0,heatmap.shape[0],heatmap.shape[1])) < 0.8:
                    cropped_img = images[i].crop(box)
                    cropped_img = cropped_img.resize([heatmap.shape[0],heatmap.shape[1]])
                    if "tile2vec" in target_name:
                        cropped_img = (torch.from_numpy(np.float64(np.array(cropped_img)) / 255).permute(2, 0, 1)).type(torch.FloatTensor)
                    all_image_crops.append(cropped_img)
                    cropped_bb.append(box)

            all_bb_box[target_layer][i] = cropped_bb
        if(return_bounding_box == True):
            utils.show_otsu_threshold(thresholded_feature_maps, thresholds)
            utils.show_bbox_on_heatmap(all_heatmaps[target_layer], all_bb_box[target_layer])  
    if return_bounding_box == True:
        return all_bb_box[target_layers[0]], all_image_crops
    else:
        del all_bb_box
        return all_image_crops



def get_attention_crops_brain_onlycropped(target_name, images, neuron_id, num_crops_per_image = 4, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg', return_bounding_box = False, model=None, preprocess=None,weights=None,box_dir=None,feature_dir=None,heat_dir=None):

#
    target_dir = os.path.join(box_dir, f'neuron_{neuron_id}')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
#
    target_dir2 = os.path.join(feature_dir, f'neuron_{neuron_id}')
    if not os.path.exists(target_dir2):
        os.makedirs(target_dir2)
#
    target_dir3 = os.path.join(heat_dir, f'neuron_{neuron_id}')
    if not os.path.exists(target_dir3):
        os.makedirs(target_dir3)

    if target_name == 'custom':
        target_model, preprocess = data_utils.get_target_model(target_name, device, model, preprocess)
    else:
        target_model, preprocess = data_utils.get_target_model(target_name, device)
    all_features = {target_layer:[] for target_layer in target_layers}
    hooks = {}
    
    transform = transforms.ToPILImage()
    
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(utils.get_mean_activation_brain(all_features[target_layer],weights))".format(target_layer)
        hooks[target_layer] = eval(command)
        
    with torch.no_grad():
        for image in images:
            if "tile2vec" not in target_name:
          
                if "clip" in target_name:  
                    features = target_model.encode_image(preprocess(image).unsqueeze(0).to(device))  # 使用encode_image提取图像特征
                else:
                    features = target_model(preprocess(image).unsqueeze(0).to(device))
            
            elif "custom" in target_name:
                features = target_model(preprocess(np.array(image)).unsqueeze(0).to(device))
            else:
                features = target_model.encode(image.unsqueeze(0).to(device))
    
    all_heatmaps = {target_layer:[] for target_layer in target_layers}

    for target_layer in target_layers:
        all_features[target_layer] = torch.cat(all_features[target_layer])
        hooks[target_layer].remove()
        
        for i in range(len(all_features[target_layer])):
            #print(all_features[target_layer][i].shape)
            if "tile2vec" in target_name:
                images[i] = Image.fromarray(np.uint8(images[i].permute(1,2,0).numpy() * 255))
            if target_layer != 'fc' and target_layer != 'encoder':
                heatmap = transform(all_features[target_layer][i][neuron_id]) 
                #print(all_features[target_layer][i])         
            
            elif target_layer == 'encoder':
                unflattend_img = torch.unflatten(all_features[target_layer][i],0,(16,16,3))
                unflattend_img = torch.permute(unflattend_img, (2,0,1))
                heatmap = ImageOps.grayscale(transform(unflattend_img))
            else:
                heatmap = transform(all_features[target_layer][i])
            heatmap = heatmap.resize([images[i].size[0],images[i].size[1]])
            heatmap = np.array(heatmap)
            all_heatmaps[target_layer].append(heatmap)
        if(return_bounding_box == True):
            utils.show_binarized_heatmap(all_heatmaps[target_layer])
    
    all_image_crops = [];
    all_bb_box = {layer : {i:[] for i in range(len(all_heatmaps[target_layer]))} for layer in target_layers}
    thresholded_feature_maps = []
    thresholds = []
    for target_layer in target_layers:
        for i, heatmap in enumerate(all_heatmaps[target_layer]): 
            thresh_val, thresh = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresholded_feature_maps.append(thresh)
            thresholds.append(thresh_val)
            bb_cor = []
            # Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                box = (x, y, x + w, y + h)
                bb_cor.append(box)
            bb_cor = sorted(bb_cor, key=functools.cmp_to_key(utils.compare))
            cropped_bb = []
            for box in bb_cor:
                if len(cropped_bb) == num_crops_per_image:
                    break
                p = 0
                good_to_add = True
                while p < len(cropped_bb):
                    if utils.IoU(box, cropped_bb[p]) <= 0.5: 
                        p += 1
                    else:
                        good_to_add = False
                        break
                if good_to_add and utils.IoU(box,(0,0,heatmap.shape[0],heatmap.shape[1])) < 0.8:

                    save_path = os.path.join(target_dir, f'image_{i}_boxed.jpg')
                    # 
                    if not os.path.exists(save_path):
                        img_with_box = images[i].copy()  # 
                        draw = ImageDraw.Draw(img_with_box)
                        #draw.rectangle(box, outline="red", width=3)  # 
                        img_with_box.save(save_path)  # 
                    else:
                        # 
                        img_with_box = Image.open(save_path)
                        draw = ImageDraw.Draw(img_with_box)
                        #draw.rectangle(box, outline="red", width=3)  # 
                        img_with_box.save(save_path)  # 

                    save_path2 = os.path.join(target_dir2, f'image_{i}_boxed.jpg')

                    feature_map = all_features[target_layer][i][neuron_id]  # shape: (H, W)
                    feature_map = feature_map.detach().cpu().numpy()  # -> numpy array
                    #
                    feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-8)
                  
                    colormap = plt.get_cmap('viridis')  # 
                    heatmap_rgb = colormap(feature_map)[:, :, :3]  # 
                
                    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)  # shape: (H, W, 3)
                 
                    feature_img = Image.fromarray(heatmap_rgb)
                 
                    feature_img = feature_img.resize(images[i].size, Image.BILINEAR)

                    if not os.path.exists(save_path2):
                        img_with_box = feature_img  # 
                        draw = ImageDraw.Draw(img_with_box)
                        #draw.rectangle(box, outline="red", width=3)  # 
                        img_with_box.save(save_path2)  # 
                    else:
                        
                        img_with_box = Image.open(save_path2)
                        draw = ImageDraw.Draw(img_with_box)
                        #draw.rectangle(box, outline="red", width=3)  # 
                        img_with_box.save(save_path2)  # 

                    save_path3 = os.path.join(target_dir3, f'image_{i}_boxed.jpg')
           
                    if heatmap.dtype != np.uint8:
                        heatmap2 = (255 * heatmap).clip(0, 255).astype(np.uint8)
                    else:
                        heatmap2=heatmap
                    if heatmap2.ndim == 2:  # 
                        heatmap2 = np.stack([heatmap2] * 3, axis=-1)  
                    heatmap_img = Image.fromarray(heatmap2)

                    if not os.path.exists(save_path3):
                        img_with_box = heatmap_img  #
                        draw = ImageDraw.Draw(img_with_box)
                        #draw.rectangle(box, outline="red", width=3)  # 
                        img_with_box.save(save_path3)  # 
                    else:
         
                        img_with_box = Image.open(save_path3)
                        draw = ImageDraw.Draw(img_with_box)
                        #draw.rectangle(box, outline="red", width=3)  # 
                        img_with_box.save(save_path3)  # 

                    cropped_img = images[i].crop(box)
                    cropped_img = cropped_img.resize([heatmap.shape[0],heatmap.shape[1]])
                    if "tile2vec" in target_name:
                        cropped_img = (torch.from_numpy(np.float64(np.array(cropped_img)) / 255).permute(2, 0, 1)).type(torch.FloatTensor)
                    all_image_crops.append(cropped_img)
                    cropped_bb.append(box)

            all_bb_box[target_layer][i] = cropped_bb
        if(return_bounding_box == True):
            utils.show_otsu_threshold(thresholded_feature_maps, thresholds)
            utils.show_bbox_on_heatmap(all_heatmaps[target_layer], all_bb_box[target_layer])  
    if return_bounding_box == True:
        return all_bb_box[target_layers[0]], all_image_crops
    else:
        del all_bb_box
        return all_image_crops

def get_attention_crops_brain_new_cropped(target_name, images, neuron_id, num_crops_per_image = 4, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg', return_bounding_box = False, model=None, preprocess=None,weights=None,box_dir=None,feature_dir=None,heat_dir=None):

    target_dir = os.path.join(box_dir, f'neuron_{neuron_id}')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    target_dir2 = os.path.join(feature_dir, f'neuron_{neuron_id}')
    if not os.path.exists(target_dir2):
        os.makedirs(target_dir2)

    target_dir3 = os.path.join(heat_dir, f'neuron_{neuron_id}')
    if not os.path.exists(target_dir3):
        os.makedirs(target_dir3)

    if target_name == 'custom':
        target_model, preprocess = data_utils.get_target_model(target_name, device, model, preprocess)
    else:
        target_model, preprocess = data_utils.get_target_model(target_name, device)
    all_features = {target_layer:[] for target_layer in target_layers}
    hooks = {}
    
    transform = transforms.ToPILImage()
    
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(utils.get_mean_activation_brain(all_features[target_layer],weights))".format(target_layer)
        hooks[target_layer] = eval(command)
        
    with torch.no_grad():
        for image in images:
            if "tile2vec" not in target_name:

                if "clip" in target_name:  
                    features = target_model.encode_image(preprocess(image).unsqueeze(0).to(device))  # 使用encode_image提取图像特征
                else:
                    features = target_model(preprocess(image).unsqueeze(0).to(device))

            elif "custom" in target_name:
                features = target_model(preprocess(np.array(image)).unsqueeze(0).to(device))
            else:
                features = target_model.encode(image.unsqueeze(0).to(device))
    
    all_heatmaps = {target_layer:[] for target_layer in target_layers}

    for target_layer in target_layers:
        all_features[target_layer] = torch.cat(all_features[target_layer])
        hooks[target_layer].remove()
        
        for i in range(len(all_features[target_layer])):

            if "tile2vec" in target_name:
                images[i] = Image.fromarray(np.uint8(images[i].permute(1,2,0).numpy() * 255))
            if target_layer != 'fc' and target_layer != 'encoder':
                heatmap = transform(all_features[target_layer][i][neuron_id]) 

            elif target_layer == 'encoder':
                unflattend_img = torch.unflatten(all_features[target_layer][i],0,(16,16,3))
                unflattend_img = torch.permute(unflattend_img, (2,0,1))
                heatmap = ImageOps.grayscale(transform(unflattend_img))
            else:
                heatmap = transform(all_features[target_layer][i])
            heatmap = heatmap.resize([images[i].size[0],images[i].size[1]])
            heatmap = np.array(heatmap)
            all_heatmaps[target_layer].append(heatmap)
        if(return_bounding_box == True):
            utils.show_binarized_heatmap(all_heatmaps[target_layer])
    
    all_image_crops = [];
    all_bb_box = {layer : {i:[] for i in range(len(all_heatmaps[target_layer]))} for layer in target_layers}
    thresholded_feature_maps = []
    thresholds = []
    for target_layer in target_layers:
        for i, heatmap in enumerate(all_heatmaps[target_layer]): 
            thresh_val, thresh = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresholded_feature_maps.append(thresh)
            thresholds.append(thresh_val)
            bb_cor = []
            # Find contours
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                box = (x, y, x + w, y + h)
                bb_cor.append(box)
            bb_cor = sorted(bb_cor, key=functools.cmp_to_key(utils.compare))
            cropped_bb = []
            for box in bb_cor:
                if len(cropped_bb) == num_crops_per_image:
                    break
                p = 0
                good_to_add = True
                while p < len(cropped_bb):
                    if utils.IoU(box, cropped_bb[p]) <= 0.5: 
                        p += 1
                    else:
                        good_to_add = False
                        break
                if good_to_add and utils.IoU(box,(0,0,heatmap.shape[0],heatmap.shape[1])) < 0.8:

                    save_path = os.path.join(target_dir, f'image_{i}_boxed.jpg')

                    if not os.path.exists(save_path):
                        img_with_box = images[i].copy()  # 
                        draw = ImageDraw.Draw(img_with_box)
                        draw.rectangle(box, outline="red", width=3)  # 
                        img_with_box.save(save_path)  # 
                    else:
                        
                        img_with_box = Image.open(save_path)
                        draw = ImageDraw.Draw(img_with_box)
                        draw.rectangle(box, outline="red", width=3)  # 
                        img_with_box.save(save_path)  # 

                    save_path2 = os.path.join(target_dir2, f'image_{i}_boxed.jpg')

                    feature_map = all_features[target_layer][i][neuron_id]  # shape: (H, W)
                    feature_map = feature_map.detach().cpu().numpy()  # -> numpy array
                
                    feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-8)
     
                    colormap = plt.get_cmap('viridis')  # 
                    heatmap_rgb = colormap(feature_map)[:, :, :3]  # 

                    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)  # shape: (H, W, 3)

                    feature_img = Image.fromarray(heatmap_rgb)

                    feature_img = feature_img.resize(images[i].size, Image.BILINEAR)

                    if not os.path.exists(save_path2):
                        img_with_box = feature_img  # 
                        draw = ImageDraw.Draw(img_with_box)
                        draw.rectangle(box, outline="red", width=3)  # 
                        img_with_box.save(save_path2)  # 
                    else:
                        # 
                        img_with_box = Image.open(save_path2)
                        draw = ImageDraw.Draw(img_with_box)
                        draw.rectangle(box, outline="red", width=3)  # 
                        img_with_box.save(save_path2)  # 
 
                    save_path3 = os.path.join(target_dir3, f'image_{i}_boxed.jpg')

                    if heatmap.dtype != np.uint8:
                        heatmap2 = (255 * heatmap).clip(0, 255).astype(np.uint8)
                    else:
                        heatmap2=heatmap
                    if heatmap2.ndim == 2: 
                        heatmap2 = np.stack([heatmap2] * 3, axis=-1)  
                    heatmap_img = Image.fromarray(heatmap2)

                    if not os.path.exists(save_path3):
                        img_with_box = heatmap_img 
                        draw = ImageDraw.Draw(img_with_box)
                        draw.rectangle(box, outline="red", width=3) 
                        img_with_box.save(save_path3)
                    else:
                        img_with_box = Image.open(save_path3)
                        draw = ImageDraw.Draw(img_with_box)
                        draw.rectangle(box, outline="red", width=3)  
                        img_with_box.save(save_path3) 
                    cropped_img = images[i].crop(box)
                    cropped_img = cropped_img.resize([heatmap.shape[0],heatmap.shape[1]])
                    if "tile2vec" in target_name:
                        cropped_img = (torch.from_numpy(np.float64(np.array(cropped_img)) / 255).permute(2, 0, 1)).type(torch.FloatTensor)
                    all_image_crops.append(cropped_img)
                    cropped_bb.append(box)

            all_bb_box[target_layer][i] = cropped_bb
        if(return_bounding_box == True):
            utils.show_otsu_threshold(thresholded_feature_maps, thresholds)
            utils.show_bbox_on_heatmap(all_heatmaps[target_layer], all_bb_box[target_layer])  
    if return_bounding_box == True:
        return all_bb_box[target_layers[0]], all_image_crops
    else:
        del all_bb_box
        return all_image_crops

def generate_sd_images_score(add_im, add_im_id, all_sd_imgs, pred_label, label_id, pipe, 
                       generator, num_images_per_prompt, num_inference_steps = 50):
    
    
    image_set = pipe(pred_label, generator = generator, 
                     num_images_per_prompt = num_images_per_prompt, num_inference_steps = num_inference_steps)

    for i in range(num_images_per_prompt):
        # Rescale image
        image = image_set.images[i]
        image = image.resize([224,224])

        all_sd_imgs.append(image)
        new_idx = len(add_im)
        add_im[new_idx] = image # Add image to list
        add_im_id[label_id].append(new_idx) # map new image indices to corresponding label_id
        
    del image_set
    torch.cuda.empty_cache()
    return add_im, add_im_id, all_sd_imgs