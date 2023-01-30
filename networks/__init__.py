# Author: Nicolas DEVAUX

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

from typing import Tuple,List,Union,Dict,Any,Callable

import networks.utils as n_utils
import networks.dataset as dset
import networks.custom_layers as cl
import networks.optimizers as noptim

from threading import Thread,Lock,Event
from time import sleep
from IPython.display import clear_output
import os
from copy import deepcopy
 


class ThreadedTrainer(Thread):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ThreadedTrainer, cls).__new__(cls)
        return cls._instance  
    
    def __init__(self,
                 num_threads:int,
                 DATA_PATH:str,
                 SAVE_RESULT_PATH:str,
                 SAVE_NETWORK_PATH:str,
                 SAVE_NON_ZERO:str,
                 line_size = 11,acc_update=100) -> None:
        super().__init__(name="ThreadedTrainer")
        
        #Handlers
        self.logger: ThreadedTrainer.Logger
        self.non_zero_saver : ThreadedTrainer.NonZeroSaver
        
        #Internal list and lock
        self._lock = Lock()
        self.threads:List[ThreadedTrainer.Trainers] = []
        self.queue = []
        self.results = {}
        
        # CONSTANTS
        self.acc_update = acc_update
        
        #Events
        self._kill = Event()
        self._available = Event()
        self.done = Event()
        
        #Paths
        self.DATA_PATH = DATA_PATH
        self.SAVE_RESULTS_PATH = SAVE_RESULT_PATH
        self.SAVE_NETWORK_PATH = SAVE_NETWORK_PATH
        
        # init
        self.logger = ThreadedTrainer.Logger(self,line_size)
        self.non_zero_saver = ThreadedTrainer.NonZeroSaver(SAVE_NON_ZERO)
        self.threads = [ThreadedTrainer.Trainers(self,"Trainer-" +str(i)) for i in range(num_threads)]
                
    def _get_next_in_queue(self) -> Union[Tuple[str,dict,bool], None]:
        print("get_next_in_queue")
        if self._kill.is_set():
            return None
        with self._lock:
            if len(self.queue) == 1:
                self._available.clear()
                return self.queue.pop(0)
            if len(self.queue) == 0:
                self._available.clear()
                return None
            return self.queue.pop(0)
        
    def _add_results(self,key,res):
        with self._lock:
            self.results[key] = res
    
    def _error(self,thread_name,buff: Union[Tuple[str,dict,bool],None],error):
        if buff is not None:
            self.add(*buff)
        self.logger.warning.append(f"{thread_name} error : {error}")
    
    def _save(self):
        with open(self.SAVE_RESULTS_PATH,"r+") as file:
            file_data:dict = json.load(file)
            file_data.update(self.results)
            file.seek(0)
            json.dump(self.results,file)
    
    def add(self,key,net,non_zero=False):
        with self._lock:
            if key in [k for k,_,_ in self.queue]:
                return
            for t in self.threads:
                if key == t.info["key"]:
                    return
            self.queue.append((key,net,non_zero))
        self._available.set()
        self.done.clear()
    
    def clear(self):
        with self._lock:
            self.queue.clear()
            self._save()
            self.results.clear()
    
    def stop(self):
        self._kill.set()
        self._available.set()
    
    def run(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for t in self.threads:
            t.start()
        self.logger.start()
        for t in self.threads:
            t.join()
        self.logger.join()
        self._save()
    class Trainers(Thread):
        def __init__(self,parent,name:Union[str,None] = None) -> None:
            super().__init__(name=name)
            self.parent :ThreadedTrainer = parent
            
            #runtime info
            self.epoch = 0
            self.info :Dict[str,Any] = {"acc_update":str(self.parent.acc_update)}
            self._done = Event()

            #constants
            self.DATA_PATH = parent.DATA_PATH
            self.SAVE_NETWORK_PATH = parent.SAVE_NETWORK_PATH
            
            #init 
            self._set_info_default()
               
        def _set_info(self,key:str,value:Any):
            if key not in self.info:
                raise KeyError(f"{key} is not a valid info key")
            self.info[key] = value
            self.parent.logger.big_update()
              
        def _set_info_default(self):
            self.info.update({
            "model":"NaN",
            "optimizer":"NaN",
            "dataset":"NaN",
            "accuracy":0,
            "device":"cpu",
            "max_epoch":0,
            "size":0,
            "last_update":0,
            "key":"NaN" 
            })
            self.parent.logger.big_update()
            self._done.set()
            for t in self.parent.threads:
                if not t._done.is_set():
                    return
            self.parent.done.set()
        
        def _train(self,key,net,non_zero) -> float:
            self._set_info("key",key)
            start = 0
            self._set_info("device","cuda" if net["use_cuda"] and torch.cuda.is_available() else "cpu")
            DEVICE = torch.device(self.info["device"])
            
            #region DataSets
            class_size,trainset,testset = dset.get_dataset(net["dataset"],self.DATA_PATH)
            dataLoader = lambda train,dataset : DataLoader(dataset, batch_size=net["batch_size"], shuffle= train, num_workers=2)
            trainLoader,testLoader =  dataLoader(True,trainset), dataLoader(False,testset)
            self._set_info("dataset",net["dataset"]["name"])
            #endregion
            
            #region Model
            model :nn.Module
            model = cl.get_model(net["model"],trainset[0][0].shape[1],class_size).to(DEVICE)
            self._set_info("size",str(n_utils.count_parameters(model)/1e6)[:6] + " M")
            self._set_info("model",net["model"]["name"])
            
            self.parent.non_zero_saver.register(key,non_zero)
            #endregion
            
            #region Training method
            criterion = nn.CrossEntropyLoss()
            optimizer = noptim.get_optimizer(net["optimizer"],model)       
            scheduler = noptim.get_scheduler(net["scheduler"],optimizer)
            self._set_info("optimizer", net["optimizer"]["name"])
    
            #endregion
            
            #region Recover
            with open(self.SAVE_NETWORK_PATH+f".INCOMPLETE.json","r") as file:
                data = json.load(file)
                if key in data:
                    model.load_state_dict(torch.load(self.SAVE_NETWORK_PATH+f".{key}_INCOMPLETE.pth"))
                    optimizer.load_state_dict(torch.load(self.SAVE_NETWORK_PATH+f".{key}_OPTIM_INCOMPLETE.pth"))
                    start = data[key]
            #endregion
            
            #region Training
            iters = len(trainLoader)
            self._set_info("max_epoch", net["epoch"])
            model.train()
            
            #Start training
            for self.epoch in range(start,net["epoch"]):
                self.parent.logger.update()
                
                for i,(inputs,labels) in enumerate(trainLoader,0):
                    inputs,labels = inputs.to(DEVICE),labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(self.epoch+ i/iters)
                
                self.parent.non_zero_saver.step(key,model)
                
                if (self.epoch % self.parent.acc_update == self.parent.acc_update-1) or self.parent._kill.is_set():
                    self._set_info("accuracy",n_utils.get_accuracy(model, testLoader,DEVICE))
                    self._set_info("last_update",self.epoch+1)
                    self._save_incomplete(key,model,optimizer,self.epoch+1)                    
                    if self.parent._kill.is_set():
                        return self.info["accuracy"]
                
                if self.epoch == 0: self.parent.logger.big_update() # To get max GPU usage update
            
            #finished        
            acc = n_utils.get_accuracy(model, testLoader,DEVICE)
            torch.save(model.state_dict(),self.SAVE_NETWORK_PATH+f"{key}.pth")
            self._remove_incomplete(key)
            
            self.parent.non_zero_saver.save()
            #endregion
            
            return acc                               
        
        def _save_incomplete(self,key:str,model:nn.Module,optimizer:torch.optim.Optimizer,epoch):
            torch.save(model.state_dict(),self.SAVE_NETWORK_PATH+f".{key}_INCOMPLETE.pth")
            torch.save(optimizer.state_dict(),self.SAVE_NETWORK_PATH+f".{key}_OPTIM_INCOMPLETE.pth")
            self.parent.non_zero_saver.save()
            with self.parent._lock:
                with open(self.SAVE_NETWORK_PATH+".INCOMPLETE.json","r+") as f:
                    data = json.load(f)
                    data[key] = epoch
                    f.seek(0,0)
                    json.dump(data,f)
                
        def _remove_incomplete(self,key:str):
            os.remove(self.SAVE_NETWORK_PATH+f".{key}_INCOMPLETE.pth")
            os.remove(self.SAVE_NETWORK_PATH+f".{key}_OPTIM_INCOMPLETE.pth")
            with self.parent._lock:
                with open(self.SAVE_NETWORK_PATH+".INCOMPLETE.json","r+") as f:
                    data:dict = json.load(f)
                    data.pop(key,None)
                    f.seek(0,0)
                    json.dump(data,f)
                
        def run(self) -> None:
            error_time_out = 60
            self.parent._available.wait()
            buffer = None
            try :
                while( not self.parent._kill.is_set()):
                    buffer = self.parent._get_next_in_queue()
                    if buffer is None:
                        self.parent._available.wait()
                        continue
                    try :
                        self._done.clear()
                        res = self._train(*buffer)
                        self.parent._add_results(buffer[0],res)
                        self._set_info_default()
                        sleep(5)
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            sleep(error_time_out)
                            error_time_out *= 2
                            self.parent._error(self.name,buffer,e)
                            continue
                        else:
                            raise e
            except Exception as e:
                self.parent._error(self.name,buffer,e)
                  
    class NonZeroSaver:
        def __init__(self,SAVE_PATH) -> None:
            self.SAVE_PATH = SAVE_PATH
            self.n0list:Dict[str,Dict[str,list]] = {}
            
        def register(self,key:str,non_zero:bool):
            if non_zero:
                self.n0list[key] = {
                    "conv_like": [],
                    "FC1": [],
                    "FC2": []
                }
                
        def step(self,key:str,model:nn.Module):
            if key in self.n0list:
                self.n0list[key]["conv_like"].append(n_utils.count_non_zero_parameters(model,"conv_like"))
                self.n0list[key]["FC1"].append(n_utils.count_non_zero_parameters(model,"FC.0"))
                self.n0list[key]["FC2"].append(n_utils.count_non_zero_parameters(model,"FC.3"))
        
        def save(self):
            with open(self.SAVE_PATH,"r+") as file:
                file_data:dict = json.load(file)
                file_data.update(self.n0list)
                file.seek(0)
                json.dump(file_data,file)

    class Logger(Thread):
        def __init__(self,trainer,line_size=11):
            Thread.__init__(self)
            
            self.line_size = line_size
            self.trainer:ThreadedTrainer = trainer
            self.warning = []
            
            #Event 
            self._update = Event()
            self._big_update = Event()
        
        def _to_str(self,info:Any) -> str:
            if isinstance(info,float):
                return f"{info:.2f}"
            return str(info) 
        
        def _line(self,call: Callable[[Any],Any]) -> str:
            infos = [self._to_str(call(i)) for i in self.trainer.threads]
            return " | ".join([j + " "*(self.line_size-len(j)) for j in infos])
        def _line2(self,call: Callable[[Any],Tuple[Any,Any]]) -> str:
            infos = [call(i) for i in self.trainer.threads]
            infos = [(self._to_str(j1),self._to_str(j2)) for j1,j2 in infos]
            return " | ".join([j1 + " "*(self.line_size-len(j1)-len(j2)) +j2 for j1,j2 in infos])
        def _line_center(self,call: Callable[[Any],Any]) :
            infos = [self._to_str(call(i)) for i in self.trainer.threads]
            return " | ".join([i.center(self.line_size) for i in infos])
        
        def _get_warning(self):
            if len(self.warning) == 0:
                return ""
            return "WARNING !!! :\n" + "\n".join(self.warning) + "\n\n\n"
        def _get_task_remaining(self) -> str:
            a = f"{len(self.trainer.queue)} tasks remaining\n" 
            
            l = [str(i+1)+". " + x[0] for i,x in enumerate(self.trainer.queue)]
            ml = max([len(i) for i in l])
            for i in range(len(l)//2):
                a += "\t" + l[i] 
                a += " "*(ml-len(l[i])+10)
                a +=  l[len(l)//2 + i + len(l)%2]+"\n"
            if len(l)%2 == 1:
                a += "\t" + l[len(l)//2] + "\n"             
            return a
        def _get_general_info(self) -> str:
            a = ""
            a += f"Accuracy updated every {self.trainer.acc_update} epochs\n"
            if torch.cuda.is_available():
                available,total = torch.cuda.mem_get_info()
                max_usage = torch.cuda.max_memory_allocated()
                a+= "GPU available: "+ self._to_str(available/1e9) + "GB out of " + self._to_str(total/1e9) + "GB\n"
                a+= "GPU max usage: "+ self._to_str(max_usage/1e9) + "GB out of " + self._to_str(total/1e9) + "GB\n"
            return a
        def _get_result(self) -> str:
            
            a = "     Model     |   Optimizer   |    Dataset    ||    optimization_parameters   ||    accuracy   \n"
            a+= "---------------|---------------|---------------||------------------------------||---------------\n"
            for key,item in self.trainer.results.items():
                data = key.split("_")
                model = data[0]
                opt = data[1]
                dataset = data[2]
                param = data[3] if len(data) == 4 else "¤"
                a += f"{model.center(15)}|{opt.center(15)}|{dataset.center(15)}||"
                a += f"{param.center(30)}||"+self._to_str(item).center(15)+"\n"

            return a        
        def _get_epoch_line(self)->str:    
            return "Epoch     || " + self._line(lambda x : x.epoch) + " ||\r"       
        def _get_thread(self)->str:
            a = ""
            a+= "Thread    || " + self._line_center(lambda x: x.name)                               + " ||\n"
            a+= "----------||-" + "-|-".join(["-"*self.line_size for i in self.trainer.threads])    + "-||\n"
            a+= "Model     || " + self._line(lambda x: x.info["model"])                             + " ||\n"
            a+= "Dataset   || " + self._line(lambda x: x.info["dataset"])                           + " ||\n"
            a+= "Optimizer || " + self._line(lambda x: x.info["optimizer"])                         + " ||\n"
            a+= "cuda      || " + self._line(lambda x: x.info["device"])                            + " ||\n"
            a+= "epoch max || " + self._line(lambda x: x.info["max_epoch"])                         + " ||\n"
            a+= "Size      || " + self._line(lambda x: x.info["size"])                              + " ||\n"
            a+= "----------||-" + "-|-".join(["-"*self.line_size for i in self.trainer.threads])    + "-||\n"
            a+= "Accuracy  || " + self._line2(
                lambda x :(self._to_str(x.info["last_update"]) + ":" ,
                           self._to_str(x.info["accuracy"])+"%"))                                   + " ||\n"
            return a
                 
        def __str__(self):
            a = self._get_warning()
            a += self._get_task_remaining()
            a += "\n\n"
            a += self._get_general_info()
            a += "\n\n"
            a += self._get_result() 
            a += "\n\n"
            a += self._get_thread()
            a += self._get_epoch_line()
            return a

        def big_update(self):
            self._big_update.set()
            self._update.set()
        def update(self):
            self._update.set()
        
        def _print(self):
            clear_output(wait=True)
            sleep(0.1)
            print(self,end="") 
            
        def run(self):
            self._print()  
            while not self.trainer._kill.is_set():
                self._update.wait()
                if self._big_update.is_set():
                    self._print()  
                    self._big_update.clear()
                else :
                    print(self._get_epoch_line(),end="")
                self._update.clear()
                sleep(0.1)
            
            for i in self.trainer.threads:
                i._done.wait(20)
            self._print()  

def network_thread(optim:dict,network_name:str,trainer:ThreadedTrainer):
    Thread(target=network_Optimizer,args=(optim,network_name,trainer)).start()

def network_Optimizer(optim:dict,network_name:str,trainer:ThreadedTrainer):
    optim_dict = optim["optim"]
    optim["networks"][network_name] 

    to_optimize = []
    #get the parameters to optimize
    for structure in ["model","dataset","optimizer","scheduler"]:
        for param, value in optim_dict[structure][optim["networks"][network_name] [structure]["type"]].items():
            to_optimize.append((structure,param,value))
    #add result
    optim[network_name] = {"accuracy":0,"best":{}}
    for structure,param,_ in to_optimize:
        optim[network_name]["best"][structure+"."+param] = optim["networks"][network_name] [structure][param]
    #get baseline
    trainer.add(network_name+"_baseline",optim["networks"][network_name] )
    trainer.done.wait()
    optim[network_name]["accuracy"] = trainer.results[network_name+"_baseline"]
    #optimize
    for structure,param,value in to_optimize:
        for i in [val for val in to_optimize if val != optim["networks"][network_name] [structure][param]]:
            network_copy = deepcopy(optim["networks"][network_name] )
            network_copy[structure][param] = i
            trainer.add(network_name+"_"+structure+"."+param.replace("_","-")+"="+i,network_copy)
        trainer.done.wait()
        for i in [val for val in to_optimize if val != optim["networks"][network_name][structure][param]]:
            new_name = network_name+"_"+structure+"."+param.replace("_","-")+"="+i
            if trainer.results[new_name] > optim[network_name]["accuracy"]:
                optim[network_name]["best"][structure+"."+param] = i
                optim[network_name]["accuracy"] = trainer.results[new_name]

            
    pass 
    