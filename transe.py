import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.init import xavier_uniform_
from torch_geometric.datasets import WordNet18RR
from torch_geometric.utils import negative_sampling


class TransE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int):
        super(TransE, self).__init__()
        self.emb_dim = emb_dim
        self.ent_embeddings = nn.Embedding(num_entities, emb_dim)
        self.rel_embeddings = nn.Embedding(num_relations, emb_dim)
        xavier_uniform_(self.ent_embeddings.weight.data)
        xavier_uniform_(self.rel_embeddings.weight.data)

    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        if tails is None:
            # compute scores for all possible tails
            head_embeddings = self.ent_embeddings(heads)
            rel_embeddings = self.rel_embeddings(relations)
            score = head_embeddings + rel_embeddings.unsqueeze(1) - self.ent_embeddings.weight.unsqueeze(0)
            score = score.norm(p=2, dim=-1).pow(2)
            return score
        else:
            # compute scores for the given tails
            head_embeddings = self.ent_embeddings(heads)
            rel_embeddings = self.rel_embeddings(relations)
            tail_embeddings = self.ent_embeddings(tails)
            score = head_embeddings + rel_embeddings - tail_embeddings
            score = score.norm(p=2, dim=-1).pow(2)
            return score

    def train_step(self, selected_optimizer: optim.Optimizer, heads: torch.Tensor, relations: torch.Tensor,
                   tails: torch.Tensor, negative_tails: torch.Tensor, margin: float):
        # compute the positive score
        positive_scores = self.forward(heads, relations, tails)
        if negative_tails is not None:
            # compute the negative score
            negative_scores = self.forward(heads, relations, negative_tails)
            # compute the loss and update the model
            curr_loss = torch.mean(torch.relu(positive_scores + margin - negative_scores))
        else:
            # compute the loss and update the model
            curr_loss = torch.mean(torch.relu(margin - positive_scores))
        selected_optimizer.zero_grad()
        curr_loss.backward()
        selected_optimizer.step()
        return curr_loss.item()


# load the WordNet18RR dataset
dataset = WordNet18RR('data/WordNet18RR')
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
# init model
model = TransE(dataset.num_entities, dataset.num_relations, dataset.emb)
# set model to training mode
model.train()
# conditional device pick
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MarginRankingLoss(margin=1.0)
# hyperparams definition
num_entities = 40943
embedding_dim = 200
learning_rate = 0.001
num_epochs = 100
print_interval = 10
margin = 1.0

# train loop
for epoch in range(num_epochs):
    running_loss = 0
    for i, data in enumerate(dataloader):
        pos_samples = data.train_pos.to(device)
        model.train()
        optimizer.zero_grad()
        pos_samples = pos_samples.to(device)
        batch_size = pos_samples.x.shape[0]

        # generate negative samples
        neg_samples = negative_sampling(pos_samples.edge_index, num_nodes=num_entities, num_neg_samples=5)

        # compute loss and perform backpropagation
        pos_scores = model(pos_samples.src, pos_samples.edge_type, pos_samples.dst)
        neg_scores = model(neg_samples[:, 0], neg_samples[:, 1], neg_samples[:, 2])
        loss = model.train_step(optimizer, pos_samples[:, 0], pos_samples[:, 1], pos_samples[:, 2], neg_samples[:, 2],
                                margin)
        running_loss += loss

        # print progress
        if (i + 1) % print_interval == 0:
            print(f"Epoch {epoch + 1}, iteration {i + 1}, loss: {running_loss / print_interval:.4f}")
            running_loss = 0

    # evaluate model after each epoch
    with torch.no_grad():
        model.eval()
        pos_samples = dataset.test_pos.to(device)
        neg_samples = dataset.test_neg.to(device)
        pos_scores = model(pos_samples[:, 0], pos_samples[:, 1], None)
        neg_scores = model(neg_samples[:, 0], neg_samples[:, 1], neg_samples[:, 2])
        hits_at_10 = torch.mean((pos_scores.unsqueeze(1) >= neg_scores.unsqueeze(0)).sum(dim=-1).float())
        print(f"Epoch {epoch + 1} evaluation: Hits@10: {hits_at_10.item():.4f}")


# save the trained model
torch.save(model.state_dict(), 'transe_wordnet18rr.pth')
