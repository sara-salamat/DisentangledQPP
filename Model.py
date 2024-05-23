import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
from transformers import AutoTokenizer, AutoModel



is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print('Warning: GPU is not available!')




class decoding_classification(nn.Module):
    def __init__(self, layers_sizes=[28,1]) -> None:
        super().__init__()
        self.layer1 = nn.Linear(layers_sizes[0], 512)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 512)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(512, layers_sizes[1])
        self.activation3 = nn.Sigmoid()
    
    def forward(self,x):

        return self.activation3(self.layer3(self.activation2(self.layer2(self.activation1(self.layer1(x))))))

class decoding_regression(nn.Module):
    def __init__(self, layers_sizes=[28,1]) -> None:
        super().__init__()
        self.layer1 = nn.Linear(layers_sizes[0], 512)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(512, layers_sizes[1])
        self.activation2 = nn.Sigmoid()
    
    def forward(self,x):

        return self.activation2(self.layer2(self.activation1(self.layer1(x))))
    
class TransformerPredictor(nn.Module):
    def __init__(self,input_size ,latent_space_size=128, content_portion_size=100,model_name=None,) -> None:
        super().__init__()
        self.latent_space_size = latent_space_size
        self.content_portion_size = content_portion_size

        
        self.encoder = AutoModel.from_pretrained(model_name)
        # self.linear_transformation = nn.Linear(input_size, 128)

        self.regression = decoding_classification(layers_sizes=[latent_space_size-content_portion_size, 1])
        self.initial_score_predictor = decoding_classification(layers_sizes=[latent_space_size-content_portion_size, 1])
        self.target_score_predictor = decoding_classification(layers_sizes=[latent_space_size-content_portion_size, 1])

    def forward(self,initial_q, initial_score, target_q, target_score, similarity_label):

        initial_q = self.encoder(**(initial_q))
        target_q = self.encoder(**(target_q))

        latentspace_initial_q = initial_q.last_hidden_state.mean(axis=1)

        latentspace_target_q = target_q.last_hidden_state.mean(axis=1)

        # content_vec_initial_q = latentspace_initial_q[:,:self.content_portion_size]

        # content_vec_target_q = latentspace_target_q[:,:self.content_portion_size]
        
        content_vec_initial_q = latentspace_initial_q[:,-self.content_portion_size:]
        content_vec_target_q = latentspace_target_q[:,-self.content_portion_size:]

        hardness_vec_initial_q = latentspace_initial_q[:,:-self.content_portion_size]
        hardness_vec_target_q = latentspace_target_q[:,:-self.content_portion_size]
        
        content_loss = nn.CosineEmbeddingLoss()(content_vec_initial_q,content_vec_target_q, similarity_label)
            


        # concat_vec = torch.cat([hardness_vec_initial_q.reshape((hardness_vec_initial_q.shape[0],1,hardness_vec_initial_q.shape[1])),
        #                         hardness_vec_target_q.reshape((hardness_vec_target_q.shape[0],1,hardness_vec_target_q.shape[1]))], dim=1)
        predicted_initial_score = self.initial_score_predictor(hardness_vec_initial_q)
        initial_score_loss = nn.MSELoss()(predicted_initial_score.reshape(-1,1), initial_score.reshape(-1,1))
        predicted_target_score = self.target_score_predictor(hardness_vec_target_q)
        target_score_loss = nn.MSELoss()(predicted_target_score.reshape(-1,1), target_score.reshape(-1,1))
  
        predicted_difference_score = self.regression(hardness_vec_initial_q-hardness_vec_target_q)
        

        actual_difference_score = (initial_score - target_score).reshape((-1,1))
        actual_labels = torch.heaviside(actual_difference_score, torch.Tensor([1]).cuda())
        difference_loss = nn.BCEWithLogitsLoss()(predicted_initial_score.reshape(-1,1)-predicted_target_score.reshape(-1,1), actual_labels)

        

        classification_loss = nn.BCELoss()(predicted_difference_score, actual_labels)

        total_loss = content_loss + 15*initial_score_loss + 15*target_score_loss+ classification_loss + 15*difference_loss

        return (total_loss, predicted_difference_score)

    
    def predict(self, initial_q, target_q):
        
        latentspace_initial_q = self.encoder(**(initial_q)).last_hidden_state.mean(axis=1)
        latentspace_target_q = self.encoder(**(target_q)).last_hidden_state.mean(axis=1)


        hardness_vec_initial_q = latentspace_initial_q[:,:-self.content_portion_size]
        hardness_vec_target_q = latentspace_target_q[:,:-self.content_portion_size]
        
        predicted_difference_score = self.regression(hardness_vec_initial_q-hardness_vec_target_q)
        # predicted_initial_score = self.initial_score_predictor(hardness_vec_initial_q)
        # predicted_target_score = self.target_score_predictor(hardness_vec_target_q)

        return predicted_difference_score
    def indiv_score(self, query):
        hardness_vec_q = self.encoder(**(query)).last_hidden_state.mean(axis=1)[:,:-self.content_portion_size]
        predicted_initial_score = self.initial_score_predictor(hardness_vec_q)
        predicted_target_score = self.target_score_predictor(hardness_vec_q)
        return predicted_initial_score, predicted_target_score
    def get_difficulty_vec(self, query):
        latent_space = self.encoder(**(query)).last_hidden_state.mean(axis=1)
        hardeness = latent_space[:,:-self.content_portion_size]
        return hardeness
    def get_content_vec(self, query):
        latent_space = self.encoder(**(query)).last_hidden_state.mean(axis=1)
        content = latent_space[:,-self.content_portion_size:]
        return content