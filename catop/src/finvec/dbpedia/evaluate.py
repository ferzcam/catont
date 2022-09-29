import torch as th
import tqdm


def evaluate_e(model, loader, filters_e, device, query_type, verbose=1):
    r = []
    rr = []
    h1 = []
    h3 = []
    h10 = []
    if verbose == 1:
        loader = tqdm.tqdm(loader)
    with th.no_grad():
        for pos, mix in loader:
            mix = mix.to(device)
            if query_type == '1p':
                logits = model.predict(mix, query_type='1p', answer_type='e')
                filter_e = filters_e[(pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2p':
                logits = model.predict(mix, query_type='2p', answer_type='e')
                filter_e = filters_e[(pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '3p':
                logits = model.predict(mix, query_type='3p', answer_type='e')
                filter_e = filters_e[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2i':
                logits = model.predict(mix, query_type='2i', answer_type='e')
                filter_e = filters_e[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '3i':
                logits = model.predict(mix, query_type='3i', answer_type='e')
                filter_e = filters_e[(pos[0, 1].item(), pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'pi':
                logits = model.predict(mix, query_type='pi', answer_type='e')
                filter_e = filters_e[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'ip':
                logits = model.predict(mix, query_type='ip', answer_type='e')
                filter_e = filters_e[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2u':
                logits = model.predict(mix, query_type='2u', answer_type='e')
                filter_e = filters_e[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'up':
                logits = model.predict(mix, query_type='up', answer_type='e')
                filter_e = filters_e[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            else:
                raise ValueError
            ranks = th.argsort(logits.squeeze(dim=0), descending=True)
            rank = (ranks == (pos[0, -1])).nonzero().item() + 1
            ranks_better = ranks[:rank - 1]
            for t in filter_e:
                if (ranks_better == t).sum() == 1:
                    rank -= 1
            r.append(rank)
            rr.append(1 / rank)
            if rank == 1:
                h1.append(1)
            else:
                h1.append(0)
            if rank <= 3:
                h3.append(1)
            else:
                h3.append(0)
            if rank <= 10:
                h10.append(1)
            else:
                h10.append(0)
    r = int(sum(r) / len(r))
    rr = round(sum(rr) / len(rr), 3)
    h1 = round(sum(h1) / len(h1), 3)
    h3 = round(sum(h3) / len(h3), 3)
    h10 = round(sum(h10) / len(h10), 3)
    print(f'#Entity#{query_type}# MRR: {rr}, H1: {h1}, H3: {h3}')
    return r, rr, h1, h3, h10


def evaluate_c(model, loader, filters_c, device, query_type, cfg):
    r = []
    rr = []
    h1 = []
    h3 = []
    h10 = []
    if cfg.verbose == 1:
        loader = tqdm.tqdm(loader)
    with th.no_grad():
        for pos, mix in loader:
            mix = mix.to(device)
            if query_type == '1p':
                logits = model.predict(mix, query_type='1p', answer_type='c')
                filter_c = filters_c[(pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2p':
                logits = model.predict(mix, query_type='2p', answer_type='c')
                filter_c = filters_c[(pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '3p':
                logits = model.predict(mix, query_type='3p', answer_type='c')
                filter_c = filters_c[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2i':
                logits = model.predict(mix, query_type='2i', answer_type='c')
                filter_c = filters_c[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '3i':
                logits = model.predict(mix, query_type='3i', answer_type='c')
                filter_c = filters_c[(pos[0, 1].item(), pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'pi':
                logits = model.predict(mix, query_type='pi', answer_type='c')
                filter_c = filters_c[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'ip':
                logits = model.predict(mix, query_type='ip', answer_type='c')
                filter_c = filters_c[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == '2u':
                logits = model.predict(mix, query_type='2u', answer_type='c')
                filter_c = filters_c[(pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            elif query_type == 'up':
                logits = model.predict(mix, query_type='up', answer_type='c')
                filter_c = filters_c[(pos[0, 2].item(), pos[0, 3].item(), pos[0, 4].item(), pos[0, 5].item(), pos[0, 6].item())]
            else:
                raise ValueError
            ranks = th.argsort(logits.squeeze(dim=0), descending=True)
            rank = (ranks == (pos[0, -1])).nonzero().item() + 1
            ranks_better = ranks[:rank - 1]
            for t in filter_c:
                if (ranks_better == t).sum() == 1:
                    rank -= 1
            r.append(rank)
            rr.append(1 / rank)
            if rank == 1:
                h1.append(1)
            else:
                h1.append(0)
            if rank <= 3:
                h3.append(1)
            else:
                h3.append(0)
            if rank <= 10:
                h10.append(1)
            else:
                h10.append(0)
    r = int(sum(r) / len(r))
    rr = round(sum(rr) / len(rr), 3)
    h1 = round(sum(h1) / len(h1), 3)
    h3 = round(sum(h3) / len(h3), 3)
    h10 = round(sum(h10) / len(h10), 3)
    print(f'#Concept#{query_type}# MRR: {rr}, H1: {h1}, H3: {h3}')
    return r, rr, h1, h3, h10
