from structured_attention_graphs.generation import generate_sag
from structured_attention_graphs.generation import generate_sag_and_subex
from structured_attention_graphs.utils import load_model_new
import os


model_list = ['resnet50', 'vgg19', 'resnet50_c1', 'resnet50_c2', 'resnet50d', 'swin_t', 'deit_s', 'pit_s', 'deit_s_distilled', 'pit_s_distilled', 'levit_256', 'convnext_t']

final_dict = {}

for modelnm in model_list:
    model = load_model_new(cuda=1, model_name=model)
    model_results = generate_sag_and_subex(input_img=img_path, model=model)
    final_dict[modelnm] = model_results



# roots = generate_sag('structured_attention_graphs/Images/peacock/n01806143_225.JPEG')
# print(roots)


# generation.generate_sag()