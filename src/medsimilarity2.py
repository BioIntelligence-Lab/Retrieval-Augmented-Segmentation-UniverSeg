from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from medsimilarity import utils
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchvision import transforms
import imagehash
import random
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ssim = SSIM().eval().to(device)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze',normalize=True).eval().to(device)
transform = transforms.Compose([transforms.ToTensor()])

def structural_similarity(img1, img2):
 
  # Ensure both images are grayscale
  if img1.mode != 'L' or img2.mode != 'L':
    img1 = img1.convert('L')
    img2 = img2.convert('L')
  # Resize to match dimensions
  img1 = img1.resize((112,112))

  img2 = img2.resize((112,112))
  # Calculate SSIM
  image1 = transform(img1).unsqueeze(0).to(device)
  image2 = transform(img2).unsqueeze(0).to(device)
  with torch.inference_mode():
    score = ssim(image1, image2)
  return score


def lpips_similarity(img1, img2):
 
  # Ensure both images are grayscale
  if img1.mode != 'RGB' or img2.mode != 'RGB':
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')
  # Resize to match dimensions
  img1 = img1.resize((112,112))

  img2 = img2.resize((112,112))
  # Calculate SSIM
  image1 = transform(img1).unsqueeze(0).to(device)
  image2 = transform(img2).unsqueeze(0).to(device)
  with torch.inference_mode():
    score = lpips(image1, image2)
  return 1 - score

def lpips_comparison(img, dataset, top_k = 50):

  img_test = Image.open(img)
  dataset_images = [Image.open(i) for i in dataset]

  matches = []
  for i, img in enumerate(dataset_images):
    score = float(lpips_similarity(img, img_test))
    matches.append([utils.get_filename(dataset[i]), score])
    
  matches = np.array(matches, dtype=object)
  return matches[np.argsort(matches[:, 1])][::-1][:top_k]

def __structural_comparison_worker(img1, img2):

  score = structural_similarity(Image.open(img1), Image.open(img2))
  return [utils.get_filename(img1), score]

def structural_comparison1(img, dataset, top_k = 50):

  img_test = Image.open(img)
  dataset_images = [Image.open(i) for i in dataset]

  matches = []
  for i, img in enumerate(dataset_images):
    score = float(structural_similarity(img, img_test))
    matches.append([utils.get_filename(dataset[i]), score])
    
  matches = np.array(matches, dtype=object)
  return matches[np.argsort(matches[:, 1])][::-1][:top_k]

def image_hash_comparison(img1,img2):

# Compare the hashes
  hamming_distance = img1 - img2
  similarity = 1.0 - (hamming_distance / (1024))  # Normalizing to a similarity score between 0 and 1
  return similarity

def hash_comparison(img, dataset, top_k = 50):

  img_test = Image.open(img)
  dataset_images = [imagehash.phash(Image.open(i)) for i in dataset]
  hash1 = imagehash.phash(img_test)
  matches = []
  for i, img in enumerate(dataset_images):
    score = image_hash_comparison(img, hash1)
    matches.append([utils.get_filename(dataset[i]), score])
  matches = np.array(matches, dtype=object)
  return matches[np.argsort(matches[:, 1])][::-1][:top_k]

def dense_vector_comparison(
  img, 
  dataset, 
  top_k = 50, 
  use_multiprocessing = True, 
  device = None
):

  if device == None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # This method is invariant to transformations
  model = SentenceTransformer('clip-ViT-B-32', device=device)
  # Lazyload images
  sentences = [Image.open(img)] + [Image.open(path) for path in dataset]
  if use_multiprocessing:
    embds = model.encode_multi_process(
      sentences, 
      model.start_multi_process_pool()
    )
  else:
    embds = model.encode(sentences)
  scores = util.paraphrase_mining_embeddings(
    embds, 
    top_k = top_k
  )
  scores = np.array(scores, dtype=object)
  scores = (scores[np.where(scores[:,1] == 0)[0]])[:,[2,0]]
  matches = []
  for idx, score in scores:
    matches += [[utils.get_filename(dataset[int(idx)-1]), score]]
  del embds,scores,model,sentences
  torch.cuda.empty_cache()
  return np.array(matches, dtype=object)

'''Combine scores from both methods'''
def combined_score(x_ssim, x_dvrs):

  return np.sqrt(x_ssim)*np.power(x_dvrs, 2)