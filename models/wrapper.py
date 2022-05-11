import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math
import yaml
import copy
#from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from .model import CLIP
import clip
from .transMIL import TransMIL
from .eval_utils import *

import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


class CLIPWrapper(pl.LightningModule):
    def __init__(self,
                 clip_config: dict,
                 transMIL_config: dict,
                 hparams: ArgumentParser
                 ):
        """A lightning wrapper for a CLIP model as specified in the paper.

        Args:
            clip_config (dict): A dictionary containing the CLIP instantiation parameters.
            transMIL_config (dict): A dictionary containing the TransMIL instantiation parameters.
            hparals (ArgumentParser): an ArgumentParser containing global hyper parameters.
        """
        super().__init__()
        self.model = CLIP(**clip_config)
        del self.model.transformer
        del self.model.visual
        self.model_pretrained, _ = clip.load("ViT-B/32", device="cpu", jit=False)
        # LM
        self.model.transformer = self.model_pretrained.transformer
        del self.model_pretrained
        # replace the visual transformer with a chosen model
        self.set_visual_model = TransMIL(**transMIL_config)
        self.model.visual = self.set_visual_model
        del self.set_visual_model
        print("OPENAI MODEL LOADED!")

        self.minibatch_size = hparams.minibatch_size
        #self.results_saving_dir = hparams.results_saving_dir

        self.automatic_optimization = False

        with open(hparams.val_carcinoma_txts, "r") as file:
            self.txts = file.read().split('\n')[:-1]

        self.txts.sort()
        print("txts length: ", len(self.txts))

    def forward(self, image, text):
        return self.model(image, text)

    # Training loss: https://github.com/openai/CLIP/issues/83
    # Mini-batching thanks to https://github.com/crowsonkb / https://twitter.com/RiversHaveWings
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()

        image, text = train_batch

        n = math.ceil(len(image) // self.minibatch_size)
        image_mbs = torch.chunk(image, n)
        text_mbs = torch.chunk(text, n)

        # calculate original statistics
        with torch.no_grad():
            ims = [F.normalize(self.model.encode_image(im), dim=1) for im in image_mbs]
            txt = [F.normalize(self.model.encode_text(t), dim=1) for t in text_mbs]
            # gather from all GPUs
            ims = self.all_gather(torch.cat(ims))
            txt = self.all_gather(torch.cat(txt))

            if len(ims.shape) == 3:
                ims = list(ims)
                txt = list(txt)
            else:
                ims = [ims]
                txt = [txt]
            #print("after st: ", ims.shape)
            image_logits = torch.cat(ims) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).type_as(image_logits).long()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            self.log_dict({'Training Loss': loss / len(ims), 'Training Accuracy': (acc_i + acc_t) / 2 / len(image) / len(ims)}, prog_bar=True)

        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        # image loss
        for j, mb in enumerate(image_mbs):
            images_tmp = copy.deepcopy(ims)
            images_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.model.encode_image(mb), dim=1)
            image_logits = torch.cat(images_tmp) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).type_as(image_logits).long()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            self.manual_backward(loss)

        # text loss
        for j, mb in enumerate(text_mbs):
            text_tmp = copy.deepcopy(txt)
            text_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.model.encode_text(mb), dim=1)
            image_logits = torch.cat(ims) @ torch.cat(text_tmp).t() * self.model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            self.manual_backward(loss)

        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))

    # validate using this function
    def validation_step(self, val_batch, idx):
        image, wsi_names, labels = val_batch
        img_embeddings = self.model.get_img_features(image)

        # SIDENOTE: computing accuracy for validation is not meaningful
        # the chance that the network gets the label correctly is minimal

        return {'img_embeddings' : img_embeddings, 'wsi_names' : wsi_names, 'labels': labels}

    def validation_epoch_end(self, val_step_outputs):
        """
        When the epoch ends, collect all the batches from the validation fold and
        get the average precision given a piece of text. Here we provide validation diagnoses
        for the case of Dysplasia. We loop over each text and then obtain a list of average precision.
        By taking the mean over these APs, we have mAP.

        The second stage consists of keeping track of the latents.

        TODO-1: Save all these results somewhere neatly.
        TODO-2: Add simple linear regressor for classification using the CLIP image embeddings.
        """
        img_embeddings = torch.cat([batch['img_embeddings'] for batch in val_step_outputs], dim=0)
        wsi_names = np.array([name for batch in val_step_outputs for name in batch['wsi_names']])
        multi_labels = torch.cat([batch['labels'] for batch in val_step_outputs], dim=0).cpu().numpy()
        APs = []
        binary_labels = None
        indices = None
        for i, txt in enumerate(self.txts):
            txt = txt.split('\n')
            tokenized = clip.tokenize(txt)[0].unsqueeze(0).to(self.device)
            txt_embedding = self.model.get_txt_features(tokenized)

            dot_similarity = img_embeddings @ txt_embedding.t()
            #scores = F.softmax(dot_similarity * model.model.logit_scale.exp(), dim=0)
            sorted, indices = torch.topk(dot_similarity.squeeze(1), k=dot_similarity.shape[0])
            indices = indices.cpu().numpy()
            sorted_names = wsi_names[indices]
            sorted_multi_labels = multi_labels[indices]

            #AP
            ap, binary_labels = compute_metrics(sorted_multi_labels)
            APs.append(ap)

            #frame = {'origin': sorted_names, 'labels': sorted_multi_labels, 'similarity': sorted.cpu().numpy()}

            #results = pd.DataFrame(data=frame)
            #results.to_csv(self.results_saving_dir+"/{}.csv".format(i), index=False)

        print("Mean Average AP: ", np.array(APs).mean())
        print("STD AP: ", np.array(APs).std())
        self.log("Validation Mean Average Precision for Dysplasia", np.array(APs).mean())

        # TSNE Plots for image embeddings
        tsne = TSNE(n_components=2, verbose=0, perplexity=20, n_iter=500)
        z = tsne.fit_transform(img_embeddings[indices].cpu().numpy())
        df = pd.DataFrame()
        df["y"] = np.array(binary_labels)
        df[df["y"]==0] = "Normal"
        df[df["y"]==1] = "Dysplasia"
        #df["y"] = binary_texts
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]

        sns.set(rc={'figure.figsize':(16,12)})
        scatter = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), s=100, palette="hls",
                        data=df).set(title="Image Embeddings T-SNE projection")
        plt.savefig("./lightning_logs/tsne_plots/tsne_{}.png".format(self.current_epoch))
        plt.close()
        #self.logger.experiment.add_image("generated_images", scatter.get_figure, self.current_epoch)


    def predict_step(self, batch, batch_idx):
        # TODO: return txt and img latents
        # DEPRECATED
        imgs, _, names = batch
        img_embeddings = self.model.get_img_features(imgs)

        return  img_embeddings, names

    def configure_optimizers(self):

        params = [
        {"params": self.model.visual.parameters(), "lr": self.hparams.vision_lr},
        {"params": self.model.transformer.parameters(), "lr": self.hparams.lm_lr}
        ]

        optimizer = torch.optim.AdamW(params, weight_decay=0.2, betas=(0.9,0.999),eps=1e-8)

        # TODO Watch: https://github.com/openai/CLIP/issues/107
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.hparams.restart_lr_after
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
