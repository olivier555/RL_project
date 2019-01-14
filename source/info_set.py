
def hash_dict(my_dict):
    """
    Hash a dictionnary
    Key ordering has normally no importance.

    :param my_dict:
    :return:
    """
    dico = my_dict.copy()
    for key in dico:
        if type(dico[key]) == list:
            dico[key] = str(dico[key])
    return hash(frozenset(dico.items()))


