import torch as th
from src.finvec.dbpedia.utils import save_obj, load_obj
from src.finvec.dbpedia.ppc import ppc


class TrainDataset(th.utils.data.Dataset):
    def __init__(self, e_dict, c_dict, data, num_ng, filters):
        super().__init__()
        self.n_entity = len(e_dict)
        self.n_concept = len(c_dict)
        self.data = data
        self.num_ng = num_ng
        self.filters = filters

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data[idx][0] == 0:
            pos = self.data[idx].unsqueeze(dim=0)
            neg_concepts = th.randint(self.n_concept, (self.num_ng, 1))
            neg = th.zeros_like(pos).repeat(self.num_ng, 1)
            neg[:2, -2] = neg_concepts[:2, 0]
            neg[:2, -1] = pos[0, -1]
            neg[2:, -1] = neg_concepts[2:, 0]
            neg[2:, -2] = pos[0, -2]
            return th.cat([pos, neg], dim=0)
        elif self.data[idx][0] == 100:
            pos = self.data[idx].unsqueeze(dim=0)
            neg_concepts = th.randint(self.n_concept, (self.num_ng // 2, 1))
            neg_entities = th.randint(self.n_entity, (self.num_ng // 2, 1))
            neg = th.zeros_like(pos).repeat(self.num_ng, 1)
            neg[:, 0] = 100
            neg[:2, -2] = neg_entities[:, 0]
            neg[:2, -1] = pos[0, -1]
            neg[2:, -1] = neg_concepts[:, 0]
            neg[2:, -2] = pos[0, -2]
            return th.cat([pos, neg], dim=0)
        elif self.data[idx][0] == 11:
            flt = self.filters['e']['1p'][(self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_entity
        elif self.data[idx][0] == 12:
            flt = self.filters['c']['1p'][(self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_concept
        elif self.data[idx][0] == 21:
            flt = self.filters['e']['2p'][(self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_entity
        elif self.data[idx][0] == 22:
            flt = self.filters['c']['2p'][(self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_concept
        elif self.data[idx][0] == 31:
            flt = self.filters['e']['3p'][(self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_entity
        elif self.data[idx][0] == 32:
            flt = self.filters['c']['3p'][(self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_concept
        elif self.data[idx][0] == 41:
            flt = self.filters['e']['2i'][(self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_entity
        elif self.data[idx][0] == 42:
            flt = self.filters['c']['2i'][(self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_concept
        elif self.data[idx][0] == 51:
            flt = self.filters['e']['3i'][(self.data[idx][1].item(), self.data[idx][2].item(), self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_entity
        elif self.data[idx][0] == 52:
            flt = self.filters['c']['3i'][(self.data[idx][1].item(), self.data[idx][2].item(), self.data[idx][3].item(), self.data[idx][4].item(), self.data[idx][5].item(), self.data[idx][6].item())]
            n = self.n_concept
        else:
            raise ValueError
        if self.data[idx][0] != 0 and self.data[idx][0] != 100:
            neg_answers = []
            query = self.data[idx][:-1]
            while len(neg_answers) < self.num_ng:
                neg_answer = th.randint(n, (1, 1))
                if neg_answer.item() in flt:
                    continue
                neg_answers.append(neg_answer)
            neg_answers = th.cat(neg_answers, dim=0)
            neg = th.cat([query.expand(self.num_ng, -1), neg_answers], dim=1)
            return th.cat([self.data[idx].unsqueeze(dim=0), neg], dim=0)


class ValidDataset(th.utils.data.Dataset):
    def __init__(self, data, num):
        super().__init__()
        self.n_candidate = num
        self.data = data[:1000]
        self.all_candidate = th.arange(num).unsqueeze(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        return self.data[idx], th.cat([pos[:-1].expand(self.n_candidate, -1), self.all_candidate], dim=1)


class TestDataset(th.utils.data.Dataset):
    def __init__(self, data, num):
        super().__init__()
        self.n_candidate = num
        self.data = data[1000:2000]
        self.all_candidate = th.arange(num).unsqueeze(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        return self.data[idx], th.cat([pos[:-1].expand(self.n_candidate, -1), self.all_candidate], dim=1)


def load_train_and_test_data(root, num_workers, e_dict, c_dict, r_dict, query_type):
    data = load_obj(root + 'input/' + query_type + '.pkl')
    train_e = ppc(data['train']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='sample')
    train_c = ppc(data['train']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='sample')
    train_filter_e = ppc(data['train_filter']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='filter')
    train_filter_c = ppc(data['train_filter']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='filter')
    test_e = ppc(data['test']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='sample')
    test_c = ppc(data['test']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='sample')
    test_filter_e = ppc(data['test_filter']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='filter')
    test_filter_c = ppc(data['test_filter']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='filter')
    valid_dataset_e = ValidDataset(test_e, len(e_dict))
    valid_dataloader_e = th.utils.data.DataLoader(dataset=valid_dataset_e, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    valid_dataset_c = ValidDataset(test_c, len(c_dict))
    valid_dataloader_c = th.utils.data.DataLoader(dataset=valid_dataset_c, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    test_dataset_e = TestDataset(test_e, len(e_dict))
    test_dataloader_e = th.utils.data.DataLoader(dataset=test_dataset_e, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    test_dataset_c = TestDataset(test_c, len(c_dict))
    test_dataloader_c = th.utils.data.DataLoader(dataset=test_dataset_c, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    return train_e, train_c, train_filter_e, train_filter_c, test_e, test_c, test_filter_e, test_filter_c, valid_dataloader_e, valid_dataloader_c, test_dataloader_e, test_dataloader_c


def load_test_data(root, num_workers, e_dict, c_dict, r_dict, query_type):
    data = load_obj(root + 'input/' + query_type + '.pkl')
    test_e = ppc(data['test']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='sample')
    test_c = ppc(data['test']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='sample')
    test_filter_e = ppc(data['test_filter']['e'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='e', flag='filter')
    test_filter_c = ppc(data['test_filter']['c'], e_dict, c_dict, r_dict, query_type=query_type, answer_type='c', flag='filter')
    valid_dataset_e = ValidDataset(test_e, len(e_dict))
    valid_dataloader_e = th.utils.data.DataLoader(dataset=valid_dataset_e, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    valid_dataset_c = ValidDataset(test_c, len(c_dict))
    valid_dataloader_c = th.utils.data.DataLoader(dataset=valid_dataset_c, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    test_dataset_e = TestDataset(test_e, len(e_dict))
    test_dataloader_e = th.utils.data.DataLoader(dataset=test_dataset_e, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    test_dataset_c = TestDataset(test_c, len(c_dict))
    test_dataloader_c = th.utils.data.DataLoader(dataset=test_dataset_c, batch_size=1, num_workers=num_workers, shuffle=False, drop_last=False)
    return test_e, test_c, test_filter_e, test_filter_c, valid_dataloader_e, valid_dataloader_c, test_dataloader_e, test_dataloader_c
