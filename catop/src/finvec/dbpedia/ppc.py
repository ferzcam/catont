import torch as th


def ppc(data, e_mapper, c_mapper, r_mapper, query_type, answer_type, flag):
    if query_type == '1p':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    for answer in data[key]:
                        ret.append([11, 0, 0, 0, 0,
                                    e_mapper[key[0]], r_mapper[key[1]], e_mapper[answer]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    for answer in data[key]:
                        ret.append([12, 0, 0, 0, 0,
                                    e_mapper[key[0]], r_mapper[key[1]], c_mapper[answer]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == '2p':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([21, 0, 0, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([22, 0, 0, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], c_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == '3p':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([31, 0, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]], e_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([32, 0, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], r_mapper[key[3]], c_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == '2i':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([41, 0, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([42, 0, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], c_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == '3i':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([51, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]], e_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([52, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[key[4]], r_mapper[key[5]], c_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == 'pi':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([61, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]], e_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([62, 0, e_mapper[key[0]], r_mapper[key[1]], r_mapper[key[2]], e_mapper[key[3]], r_mapper[key[4]], c_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == 'ip':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([71, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]], e_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([72, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]], c_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == '2u':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([81, 0, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], e_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([82, 0, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], c_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == 'up':
        if answer_type == 'e':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])].append(e_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([91, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]], e_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        elif answer_type == 'c':
            if flag == 'filter':
                ret = {}
                for key in data:
                    ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])] = []
                    for answer in data[key]:
                        ret[(e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]])].append(c_mapper[answer])
            elif flag == 'sample':
                ret = []
                for key in data:
                    ret.append([92, 0, e_mapper[key[0]], r_mapper[key[1]], e_mapper[key[2]], r_mapper[key[3]], r_mapper[key[4]], c_mapper[data[key]]])
                ret = th.tensor(ret)
            else:
                raise ValueError
        else:
            raise ValueError

    elif query_type == 'ot':
        ret = []
        for c_1, c_2 in data.values.tolist():
            ret.append([0, 0, 0, 0, 0, 0, c_mapper[c_1], c_mapper[c_2]])
        return th.tensor(ret)

    elif query_type == 'is':
        ret = []
        for e, c in data.values.tolist():
            ret.append([100, 0, 0, 0, 0, 0, e_mapper[e], c_mapper[c]])
        return th.tensor(ret)

    else:
        raise ValueError

    return ret
