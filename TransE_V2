import torch
import torch.nn.functional as F
from torch_geometric.nn import Embedding, TransE
from torch_geometric.data import DataLoader

from torchnlp.wordnet import WordNet18RR


# Load the Wordnet18RR dataset
wn18rr = WordNet18RR()

# Define the dimension of the embedding space and the number of negative samples per positive sample
embed_dim = 200
num_negative_samples = 5
hidden_dim = 100  # Dimension of the hidden layer

# Define the TransE with an additional hidden layer
class KnowledgeGraph(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim, hidden_dim):
        super(KnowledgeGraph, self).__init__()
        self.entity_embedding = Embedding(num_entities, embed_dim)
        self.relation_embedding = Embedding(num_relations, embed_dim)
        self.hidden_layer = torch.nn.Linear(embed_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, 1)
        self.model = TransE(num_entities, embed_dim)

    def forward(self, pos_triplets, neg_triplets):
        # Embed the entities and relations
        pos_embedded_head = self.entity_embedding(pos_triplets[:, 0])
        pos_embedded_tail = self.entity_embedding(pos_triplets[:, 2])
        pos_embedded_relation = self.relation_embedding(pos_triplets[:, 1])
        neg_embedded_head = self.entity_embedding(neg_triplets[:, 0])
        neg_embedded_tail = self.entity_embedding(neg_triplets[:, 2])
        neg_embedded_relation = self.relation_embedding(neg_triplets[:, 1])

        # Apply the hidden layer activation function to the embeddings
        pos_embedded_head = F.relu(self.hidden_layer(pos_embedded_head))
        pos_embedded_relation = F.relu(self.hidden_layer(pos_embedded_relation))
        pos_embedded_tail = F.relu(self.hidden_layer(pos_embedded_tail))
        neg_embedded_head = F.relu(self.hidden_layer(neg_embedded_head))
        neg_embedded_relation = F.relu(self.hidden_layer(neg_embedded_relation))
        neg_embedded_tail = F.relu(self.hidden_layer(neg_embedded_tail))

        # Calculate the scores for the positive and negative triplets
        pos_scores = self.output_layer(pos_embedded_head + pos_embedded_relation - pos_embedded_tail)
        neg_scores = self.output_layer(neg_embedded_head + neg_embedded_relation - neg_embedded_tail)

        # Calculate the loss using the margin ranking loss
        loss = F.margin_ranking_loss(pos_scores, neg_scores, torch.Tensor([1]))

        return loss

# Initialize the TransE model, optimizer, and early stopping criteria
model = KnowledgeGraph(wn18rr.num_entities, wn18rr.num_relations, embed_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters())
early_stopper = EarlyStopping(patience=5)

# Define the dataloader for batch training
train_loader = DataLoader(wn18rr.train_data, batch_size=128, shuffle=True)

# Train the model
for epoch in range(100):
    epoch_loss = 0

    for batch in train_loader:
        pos_triplets = batch['pos']
        neg_triplets = wn18rr.sample_negatives(pos_triplets, num_negative_samples)

        loss = model(pos_triplets, neg_triplets)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss /= len(train_loader)

    print('Epoch:', epoch, 'Loss:', epoch_loss)

    if early_stopper.check_early_stopping(epoch_loss) == True:
        print('Stopping early')
        break
