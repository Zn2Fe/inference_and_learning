import torch

#region printers
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
             
#endregion

#region Model
def get_accuracy(model, testloader, DEVICE):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        model.to(DEVICE)
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total   
#endregion