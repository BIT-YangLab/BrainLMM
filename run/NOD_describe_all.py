import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import csv
import utils
import data_utils
import DnD_models
import scoring_function
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from transformers import BlipProcessor, BlipForConditionalGeneration
current_subject=8
for current_subject in range(1,10):
    num_chunks = 1
    for row_now in range(num_chunks):
        probe="NOD"
        target_name = 'clip_resnet50'                   #clip_resnet50            clip_vit-b32
        target_layer = 'visual.attnpool'                 #visual.attnpool          visual        
        #current_subject=1
        roi='all'
        d_probe = 'coco'
        batch_size = 200
        device = 'cuda:3'
        pool_mode = 'avg'
        ids_to_check='all'
        results_dir = f'/Your_path/BrainLMM/run/results/{probe}/subj{current_subject}'
        saved_acts_dir = f'/Your_path/BrainLMM/run/activation/{probe}/subj{current_subject}'
        num_images_to_check = 10   
        blip_batch_size = 10
        clip_name = 'ViT-B/16'      
        tag = f"{roi}"
        # loading Voxal weights
        print("start load")
        current_roi=0
        current_subject = int(current_subject)
        current_roi = int(current_roi)
        #clip_rn_50
        if target_name=='clip_resnet50': 
            weights = np.load(f'/Your_path/subj{current_subject}/34_session/weights_rn50-NOD-new_whole_brain.npy')
        #clip_vit-b32
        elif target_name=='clip_vit-b32':
            weights=np.load(f'/Your_path/subj{current_subject}/34_session/weights_ViT-NOD-new_whole_brain.npy')

        print("weights: ",weights.shape)
        weights=weights.T
        weights=torch.from_numpy(weights).to(device)
        weights=weights.to(dtype=torch.float32)
        weights=weights.T
        print("final weights: ",weights.shape)
        print("finish load")
        total_ids = weights.shape[1] 
        
        chunk_size = total_ids // num_chunks

        start = (row_now) * chunk_size
        end = (row_now + 1) * chunk_size if row_now < num_chunks - 1 else total_ids
        ids_to_check = [i for i in range(start, end)]

        #print(ids_to_check)
        print(len(ids_to_check))

        # Load BLIP model
        BLIP_PATH = '/Your_path/model_base_capfilt_large.pth'
        processor = BlipProcessor.from_pretrained("/Your_path/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("/Your_path/BLIP-main/blip-image-captioning-base").to(device) 
        pretrained_dict = torch.load(BLIP_PATH)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()

        # Create results folder
        results_path = utils.create_layer_folder(results_dir = results_dir, base_dir = "/Your_path/BrainLMM", target_name = target_name, 
                                d_probe = d_probe, layer = target_layer, tag = tag)
        # Get activations
        target_save_name = utils.get_save_names_brain(target_name = target_name,
                                            target_layer = target_layer, d_probe = d_probe,
                                            pool_mode=pool_mode, base_dir = '/Your_path/BrainLMM', saved_acts_dir = saved_acts_dir,roi=roi)
        utils.save_activations_brain(target_name = target_name, target_layers = [target_layer],
                                d_probe = d_probe, batch_size = batch_size, device = device,
                                pool_mode=pool_mode, base_dir = '/Your_path/BrainLMM', saved_acts_dir = saved_acts_dir,weights=weights,roi=roi)
        target_feats = torch.load(target_save_name, map_location='cpu')
        pil_data = data_utils.get_data(d_probe)
        # Find top activating images
        top_vals, top_ids = torch.topk(target_feats, k=num_images_to_check, dim=0)
        all_imgs = []
        all_img_ids = {neuron_id:[] for neuron_id in ids_to_check}

        # Find top activating image crops
        for t, orig_id in enumerate(ids_to_check):
            print("Cropping for Neuron {}/{}".format(t+1,len(ids_to_check)))
            activating_images = []
            for i, top_id in enumerate(top_ids[:, orig_id]):
                im, label = pil_data[top_id]
                im = im.resize([375,375])
                all_img_ids[orig_id].append(len(all_imgs))
                all_imgs.append(im)
                activating_images.append(im)
            cropped_images = []
            if(target_layer != 'fc'and target_layer!='visual.attnpool' and target_layer!='visual' and 'classifier' not in target_layer):
                cropped_images = DnD_models.get_attention_crops_brain(target_name, activating_images, orig_id, num_crops_per_image = 4, target_layers = [target_layer], device = device,weights=weights)
            for img in cropped_images:
                all_img_ids[orig_id].append(len(all_imgs))
                all_imgs.append(img)

        # Get target activations with D_probe + D_cropped
        target_feats = utils.get_target_activations_brain(target_name, all_imgs, [target_layer],device=device,weights=weights)
        # Find top activating images
        top_vals, top_ids = torch.sort(target_feats, dim=0, descending = True)
        comp_words = {orig_id : [] for orig_id in ids_to_check}
        top_images = {orig_id:[] for orig_id in ids_to_check}

        # Step 2 - Generate Candidate Concepts
        for neuron_num, orig_id in enumerate(ids_to_check):
            print("Neuron: {} ({}/{})".format(orig_id, neuron_num+1, len(ids_to_check)))

            # Plot and save highest activating images
            fig, images, top_images = utils.get_top_images(orig_id, top_ids, top_images, 
                                                        all_imgs, all_img_ids, num_images_to_check, 
                                                        blip_batch_size)#, convert_from_np = True
            utils.save_activating_fig(fig, results_path, orig_id)
            
            # Generate and simplify BLIP Captions
            #descriptions是每个神经元的10个blip生成的caption

            descriptions = DnD_models.blip_caption(model, processor, images, blip_batch_size, device)
            save_descriptions = os.path.join(results_path,"all_potential_description.csv")
            os.makedirs(os.path.dirname(save_descriptions), exist_ok=True)
            with open(save_descriptions, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([orig_id, descriptions])
        import pandas as pd
        import ast
        import nltk
        from collections import defaultdict, Counter
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        df=pd.read_csv(save_descriptions,header=None)  
        df = df[df[0].astype(int).isin(ids_to_check)]
        print("Total rows in df:", len(df))
        nltk.data.path.append('/Your_path/BrainLMM/umap/package_nltk')
        lemmatizer = WordNetLemmatizer()
        voxel_noun_lists = []
        noun_to_voxels = defaultdict(set)
        voxel_noun_rows = [] 
        for idx, row in df.iterrows():
            voxel_id = row[0]
            captions = ast.literal_eval(row[1])  
            noun_set = set()
            for caption in captions:
                words = word_tokenize(caption)
                blacklist = {"essential","essentials","playing","eating","jumping","player","group","kissing"}
                merged_words = []
                skip = False
                for i in range(len(words)):
                    if skip:
                        skip = False
                        continue
                    if i < len(words) - 1 and words[i].lower() == "hot" and words[i + 1].lower() in {"dog","dogs"}:
                        merged_words.append("hot dog")
                        skip = True
                    else:
                        word_lower = words[i].lower()
                        if word_lower not in blacklist:
                            if word_lower == "soccer":
                                merged_words.append("soccer player")
                            elif word_lower == "tennis":
                                merged_words.append("tennis player")
                            elif word_lower == "baseball":
                                merged_words.append("baseball player")
                            elif word_lower == "basketball":
                                merged_words.append("basketball player")
                            else: 
                                merged_words.append(words[i])
                tagged = nltk.pos_tag(merged_words)
                for word, tag in tagged:
                    if tag.startswith("NN"): 
                        word_clean = word.lower()
                        word_lemma = lemmatizer.lemmatize(word_clean, pos='n') 
                        noun_set.add(word_lemma)
            voxel_noun_lists.append(noun_set)
            row_output = [voxel_id] + sorted(noun_set)
            voxel_noun_rows.append(row_output)
            for noun in noun_set:
                noun_to_voxels[noun].add(voxel_id)

        noun_csv_save_path = os.path.join(results_path,"voxel_noun_list.csv")
        save_dir = os.path.dirname(noun_csv_save_path)
        os.makedirs(save_dir, exist_ok=True)
        seen_voxel_ids = set()

        if os.path.exists(noun_csv_save_path):
            with open(noun_csv_save_path, "r", encoding="utf-8") as f:
                for line in f:
                    voxel_id = line.strip().split(",")[0]
                    seen_voxel_ids.add(voxel_id)

        with open(noun_csv_save_path, "a", encoding="utf-8") as f:
            for row in voxel_noun_rows:
                voxel_id = row[0]
                if voxel_id in seen_voxel_ids:
                    continue
                seen_voxel_ids.add(voxel_id)
                f.write(",".join(map(str, row)) + "\n")
        print(f"Per-voxel noun list CSV saved to: {noun_csv_save_path}")

        # 统计名词出现在多少个 voxel 中
        noun_voxel_counts = {noun: len(voxels) for noun, voxels in noun_to_voxels.items()}
        # 前50个出现最频繁的名词
        top_50_nouns = Counter(noun_voxel_counts).most_common(50)
        # 打印结果
        for noun, count in top_50_nouns:
            print(f"{noun:20s} appears in {count} voxels")

        #### Step 3 – Fine-tune for Stable Diffusion ####
        from PIL import Image
        model_id = "/Your_path/diffusion/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        generator = torch.Generator(device=device).manual_seed(0)
        pipe = pipe.to(device)


    
        all_csv_lines = []



        #df_ids = pd.read_csv(noun_path, header=None, engine="python", dtype=str)
        noun_path=noun_csv_save_path

        with open(noun_path, 'r') as f:
            lines = f.readlines()
        max_cols = max(len(line.strip().split(',')) for line in lines)
        column_names = list(range(max_cols))

        df_ids = pd.read_csv(noun_path, header=None, names=column_names, engine="python", dtype=str)
        df_ids = df_ids[df_ids[0].astype(int).isin(ids_to_check)]
        print("\nStep 3: Best Concept Selection")




        comp_words = {}
        for _, row in df_ids.iterrows():
            neuron_id = int(row[0])
            nouns = [word.strip() for word in row[1:].dropna().astype(str) if word.strip() != ""]
            comp_words[neuron_id] = nouns


        replace_set = ['design','designs','graphic','graphics']
        for orig_id in ids_to_check:
            comp_words[orig_id] = [concept.lower() for concept in comp_words[orig_id]]
            for i, word in enumerate(comp_words[orig_id]):
                if word[-1] == '.':
                    comp_words[orig_id][i] = word[:-1]#
                if word.split()[-1] in replace_set:
                    new_concept = word + ' background'#
                    comp_words[orig_id].append(new_concept)
            comp_words[orig_id] = list(set(comp_words[orig_id]))#

 
        all_final_results = {neuron_id : [] for neuron_id in ids_to_check}


        num_images_per_prompt = 10   
        top_K_param = 10            
        beta_images_param = 5       
        scoring_func = 'topk-sq-mean'
        sd_prompt = 'One realistic image of {}'  
        num_inference_steps = 50     


        #
        for list_id, orig_id in enumerate(ids_to_check):
            print("Neuron {} ({}/{})".format(orig_id, list_id + 1, len(ids_to_check)))
     
            word_list = comp_words[orig_id]
  
            labels_to_check = len(word_list)
       
            add_im = {}
         
            add_im_id = {}
         
            all_sd_imgs = []


            for label_id in range(labels_to_check):
                label = word_list[label_id]
                label_folder = os.path.join("/Your_path/BrainLMM/prove/data", label.replace(" ", "_"))
                os.makedirs(label_folder, exist_ok=True)

                existing_images = [f for f in os.listdir(label_folder) if f.endswith(".png")]
                images_needed = num_images_per_prompt - len(existing_images)

                pred_label = sd_prompt.format(label)
                add_im_id[label_id] = []

                # 
                for i in range(min(num_images_per_prompt, len(existing_images))):
                    img_path = os.path.join(label_folder, existing_images[i])
                    img = Image.open(img_path).convert("RGB")
                    #
                    all_sd_imgs.append(img)
                    new_idx = len(add_im)  # 
                    #
                    add_im[new_idx] = img
                    #
                    add_im_id[label_id].append(new_idx)

                #
                if images_needed > 0:
                    add_im, add_im_id, all_sd_imgs = DnD_models.generate_sd_images_score(
                        add_im, add_im_id, all_sd_imgs, 
                        pred_label, label_id, pipe, generator,
                        images_needed, num_inference_steps
                    )
                    start_idx = len(existing_images)
                    for i in range(images_needed):
                        img = all_sd_imgs[-images_needed + i]
                        img.save(os.path.join(label_folder, f"{start_idx + i}.png"))



            # Concept Scoring
            target_feats = utils.get_target_activations_brain(target_name, all_sd_imgs, [target_layer],weights=weights,device=device)
            ranks, highest_activating = utils.rank_images(target_feats, orig_id, labels_to_check,
                                                            add_im_id, add_im, top_K_param)
            clip_weight = scoring_function.compare_images(top_images[orig_id], highest_activating, clip_name, 
                                                            device, target_name, top_K_param)
            top_avg_topk = scoring_function.get_score(ranks, mode = scoring_func, hyp_param = beta_images_param)

            top_avg_comb = []
            for i in range(len(clip_weight)):
                concept_rank = len(top_avg_topk) - scoring_function.find_by_last(top_avg_topk, clip_weight[i][1])
                weight = clip_weight[i][0]
                concept_score = concept_rank * weight
                top_avg_comb.append((concept_score, clip_weight[i][1]))
            top_avg_comb.sort(reverse = True)
            # 
            scored_labels_str = []
            for score, idx in top_avg_comb:
                label = word_list[idx]
                scored_labels_str.append(f"{label}({score:.3f})")
            # 
            line = f"{orig_id}," + ",".join(scored_labels_str)
            print(line)
            # 
            all_csv_lines.append(line)
        #
        save_path = os.path.join(results_path,"score_label.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a") as f:
            for l in all_csv_lines:
                f.write(l + "\n")
        # 
        import torch, gc, sys
        del target_feats
        del model, pipe, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        #
        gc.collect()
        #
        sys.stdout.flush()
        sys.stderr.flush()


    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import Counter
    plt_path = os.path.join(results_path,"top50.png")
    #
    file_path = save_path
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    #
    first_nouns = []
    for line in lines:
        parts = line.split(',')  #
        if len(parts) > 1:
            noun_with_score = parts[1]  #
            noun = noun_with_score.split('(')[0].strip()  #
            first_nouns.append(noun)

    # 
    freq_counter = Counter(first_nouns)
    # 
    sorted_items = freq_counter.most_common(50)  #
    words, counts = zip(*sorted_items)  # 
    #
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Frequency")
    plt.title(f"Top-50 Noun in {roi}")
    plt.tight_layout()
    plt.savefig(plt_path, dpi=300)
    plt.close()
