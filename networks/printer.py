def pd_dict_to_string(pd_dict,model)->str:
    out = f"Network: {pd_dict['model']}, {pd_dict['dataset']}, {pd_dict['optimizer']}"
    out += f" using : {sum(p.numel() for p in model.parameters())/1000000} M parameters"
    return out    
    

def dict_printer(dict_in,o:int):
    sizeMax = max([len(k) for k in dict_in.keys()])
    for k,v in dict_in.items():
        if isinstance(v,dict):
            print(o*"\t" +f"{k} :")
            dict_printer(v,o+1)
        elif isinstance(v,list):
            print(o*"\t" +f"{k} :")
            list_printer(v,o+1)
        else:
            k = k + (sizeMax - len(k))*" " if len(k) < 20 else k
            print(o*"\t" +f"{k} : {v}")
    
def list_printer(list_in,o):
    for i,v in enumerate(list_in):
        if isinstance(v,dict):
            print(o*"\t" +f"{v} :")
            dict_printer(v,o+1)
        elif isinstance(v,list):
            print(o*"\t" +f"{v} :")
            list_printer(v,o+1)
        else:
            print(o*"\t" +f"{i} : {v}")
             