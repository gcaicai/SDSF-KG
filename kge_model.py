import torch
import torch.nn as nn
import torch.nn.functional as F


class KGEModel(nn.Module):
    def __init__(self, args, model_name):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.device = args.gpu
        self.embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim]).to(self.device)
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]),
            requires_grad=False
        ).to(self.device)

        # ConvR parameters
        self.conv_convr = nn.Conv2d(in_channels=1, out_channels=args.convr_num_filters,
                              kernel_size=(args.convr_kernel_size, args.hidden_dim)).to(self.device)
        self.fc_convr = nn.Linear(args.convr_num_filters, args.hidden_dim).to(self.device)

    def forward(self, sample, relation_embedding, entity_embedding, neg=True):
        if not neg:
            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        else:
            head_part, tail_part = sample
            batch_size = head_part.shape[0]

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if tail_part == None:
                tail = entity_embedding.unsqueeze(0)
            else:
                negative_sample_size = tail_part.size(1)
                tail = torch.index_select(
                    entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'ConvR': self.ConvR
        }

        score = model_func[self.model_name](head, relation, tail)

        return score

    def TransE(self, head, relation, tail):
        score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail):
        score = (head * relation) * tail
        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def ConvR(self, head, relation, tail):
        x = torch.cat([head, relation], dim=1)
        x = x.unsqueeze(1)

        conv_out = self.conv_convr(x)
        conv_out = F.relu(conv_out)
        conv_out = conv_out.squeeze(2).squeeze(2)

        v = self.fc_convr(conv_out)
        v.unsqueeze_(1)

        score = torch.sum(v * tail, dim=2)

        return score
