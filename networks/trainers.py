


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

from typing import Tuple,List,Union,Dict,Any,Callable

import networks.utils as n_utils
import networks.dataset as dset
import networks.custom_layers as n_cl
import networks.optimizers as n_optim
from networks.network import Network

from threading import Thread,Lock,Event
from time import sleep
from IPython.display import clear_output
import os
from copy import deepcopy


class ThreadedTrainer():
    """Threaded trainer class, used to train multiple networks at the same time

    This class may only be instanciated once, each new instance will delete previous ones
    
    Raises:
        e: Raises the exception if the training fails for any reason
        e: Raises the exception if the training fails for Cuda out of memory error
    """
    
        
    __instance = None
    
    #region constructors and destructor
    def __new__(cls, *args, **kwargs):
        if cls.__instance is not None:
            print("Too avoid having too many threads, old ThreadedTrainer will be deleted")
            del cls.__instance
            print("Creating a new ThreadedTrainer")
        cls.__instance = super(ThreadedTrainer, cls).__new__(cls)
        return cls.__instance
    
    def __init__(self,
                 num_threads:int,
                 DATA_PATH:str,
                 SAVE_RESULT_PATH:str,
                 SAVE_NETWORK_PATH:str,
                 SAVE_NON_ZERO_PATH:Union[str,None] = None,
                 line_size = 11,
                 saving_step=100) -> None:
        """An instance of the ThreadedTrainer class

        Args:
            num_threads (int): Number of threads to use
            DATA_PATH (str): Path to the data folder (must already contains all datasets)
            SAVE_RESULT_PATH (str): Path to the files where the results will be saved (json format)
            SAVE_NETWORK_PATH (str): Path to the folder where the networks will be saved, must contains a file : .INCOMPLETE.json
            SAVE_NON_ZERO (str): Path to the file where the non zero weights will be saved (json format)
            line_size (int, optional): Size of the Thread table. Defaults to 11. 
            saving_step (int, optional): number of epoch before taking a snapshot of the network . Defaults to 100.
        """
        #region Logger
        self._line_size = line_size
        self.warning = []
        self._update_event = Event()
        self._big_update_event = Event()
        #endregion
        #region Internal list and lock
        self._lock = Lock()
        self.threads:List[ThreadedTrainer.Trainers] = []
        self.queue: List[Tuple[str,Network,bool]] = []
        self.results: Dict[str,float] = {}
        self._non_zero: Dict[str,Dict[str,List[int]]] = {}
        self._non_zero_buffer: Dict[str,Dict[str,List[int]]] = {}
        self._incomplete: Dict[str,int] = {}
        #endregion
        #region Events
        self._kill = Event()
        self._available = Event()
        self.done = Event()
        #endregion
        #region CONSTANTS
        self.saving_step = saving_step
        self.DATA_PATH = DATA_PATH
        self.SAVE_NON_ZERO = SAVE_NON_ZERO_PATH
        self.SAVE_NETWORK_PATH = SAVE_NETWORK_PATH
        self.SAVE_RESULT_PATH = SAVE_RESULT_PATH
        #endregion
        #region init
        self.threads = [ThreadedTrainer.Trainers(self,"Trainer-" +str(i)) for i in range(num_threads)]
        with open(self.SAVE_RESULT_PATH,"r") as f:
            self.results = json.load(f)
        with open(self.SAVE_NETWORK_PATH+".INCOMPLETE.json","r") as f:
            self._incomplete = json.load(f)
        if self.SAVE_NON_ZERO is not None:
            with open(self.SAVE_NON_ZERO,"r") as f:
                self._non_zero = json.load(f)
        for t in self.threads:
            t.start()
        #endregion
        print("Trainer ready")
            
    def __del__(self):
        self._kill.set()
        self._available.set()
        for t in self.threads:
            t.join()
        self.threads.clear()  
        ThreadedTrainer.__instance = None
        print("Trainer killed")
    #endregion
    
    #region thread methods            
    def _get_next_in_queue(self) -> Union[Tuple[str,Network,bool], None]:
        """Private method, used to get the next element in the queue

        Returns:
            Union[Tuple[str,dict,bool], None]: The next element in the queue, or None if the queue is empty
        """
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
    def _error(self,thread_name,buff: Union[Tuple[str,Network,bool],None],error):
        """Private method, used by threads to report an error

        Args:
            thread_name (_type_): own name
            buff (Union[Tuple[str,dict,bool],None]): the model that was being trained when the error occured
            error (_type_): the error that occured
        """
        if buff is not None:
            self.add(*buff)
        self.warning.append(f"{thread_name} error : {error}")
        self._big_update()
    def _big_update(self):
        """Private method, used to notify the progress method to update everything
        """
        self._big_update_event.set()
        self._update_event.set()
    def _update(self):
        """Private method, used to notify the epoch of the threads"""
        self._update_event.set()    
    #endregion
    
    #region network access methods
    def _get_network(self,key:str,model:nn.Module,optimizer:torch.optim.Optimizer) -> int:
        """Private method, used to load a network from previous incomplete training

        Args:
            key (str): Name of the network
            model (nn.Module): model to load
            optimizer (torch.optim.Optimizer): optimizer to load

        Returns:
            int: the epoch at which the training was stopped
        """
        if key in self._incomplete:
            model.load_state_dict(torch.load(self.SAVE_NETWORK_PATH+f".{key}_INCOMPLETE.pth"))
            optimizer.load_state_dict(torch.load(self.SAVE_NETWORK_PATH+f".{key}_OPTIM_INCOMPLETE.pth"))
            return self._incomplete[key]
        return 0
    def _save_incomplete(self,key:str,model:nn.Module,optimizer:torch.optim.Optimizer,epoch:int):
        """Private method, used to take a snapshot of a network during training 

        Args:
            key (str): Name of the network
            model (nn.Module): network to save
            optimizer (torch.optim.Optimizer): optimizer to save 
            epoch (int): current epoch
        """
        torch.save(model.state_dict(),self.SAVE_NETWORK_PATH+f".{key}_INCOMPLETE.pth")
        torch.save(optimizer.state_dict(),self.SAVE_NETWORK_PATH+f".{key}_OPTIM_INCOMPLETE.pth")
        with self._lock:
            self._incomplete[key] = epoch
            with open(self.SAVE_NETWORK_PATH+".INCOMPLETE.json","w") as f:
                json.dump(self._incomplete,f,indent=4)
    def _save(self,key:str,model:nn.Module,accuracy:float):
        """Private method, used to save a network after training, will delete the incomplete snapshot

        Args:
            key (str): name of the network
            model (nn.Module): network to save
            accuracy (float): accuracy of the network
        """
        torch.save(model.state_dict(),self.SAVE_NETWORK_PATH+f"{key}.pth")       
        with self._lock:
            self.results[key] = accuracy
            self._incomplete.pop(key,None)
            with open(self.SAVE_RESULT_PATH,"w") as f:
                json.dump(self.results,f,indent=4)
            with open(self.SAVE_NETWORK_PATH+".INCOMPLETE.json","w") as f:
                json.dump(self._incomplete,f,indent=4)
        os.remove(self.SAVE_NETWORK_PATH+f".{key}_INCOMPLETE.pth")
        os.remove(self.SAVE_NETWORK_PATH+f".{key}_OPTIM_INCOMPLETE.pth")
    #endregion
    
    #region non zero access methods
    def _add_non_zero(self,key:str,model:nn.Module):
        """Private method, save the current number of non zero parameters of S-[FC-CONV-LOCAL] networks to a buffer (should be called at the end of each epoch)

        Args:
            key (str): name of the network
            model (nn.Module): network 
        """
        with self._lock:
            if key not in self._non_zero_buffer:
                self._non_zero_buffer[key] = {
                    "conv_like":[],
                    "FC1":[],
                    "FC2":[]
                }
        self._non_zero_buffer[key]["conv_like"].append(n_utils.count_non_zero_parameters(model,"conv_like"))
        self._non_zero_buffer[key]["FC1"].append(n_utils.count_non_zero_parameters(model,"FC.FC1"))
        self._non_zero_buffer[key]["FC2"].append(n_utils.count_non_zero_parameters(model,"FC.FC2"))
    def _save_non_zero(self,key:str):
        """Private method, save number of non zero parameters to file, empty the buffer"""
        with self._lock:
            if not key in self._non_zero:
                self._non_zero[key] = {
                    "conv_like":[],
                    "FC1":[],
                    "FC2":[]
                }
            if key in self._non_zero_buffer:
                self._non_zero[key]["conv_like"].extend(self._non_zero_buffer[key]["conv_like"])
                self._non_zero[key]["FC1"].extend(self._non_zero_buffer[key]["FC1"])
                self._non_zero[key]["FC2"].extend(self._non_zero_buffer[key]["FC2"])
                self._non_zero_buffer.pop(key,None)
            if self.SAVE_NON_ZERO is not None:
                with open(self.SAVE_NON_ZERO,"w") as f:
                    json.dump(self._non_zero,f,indent=4)
    #endregion
    
    #region public methods     
    def add(self,key:str,net:Network,non_zero=False):
        """Add a network to the queue

        Args:
            key (str): name of the network
            net (Nework): description of the network
            non_zero (bool, optional): Wether to save the number of non zero parameters for each layer (only for Shallow Networks). Defaults to False.
        """
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
        """Clear the queue"""
        with self._lock:
            self.queue.clear()
    def progress(self):
        """Print the progress of the training with update"""
        self._big_update()
        while True:
            self._update_event.wait()
            if self._big_update_event.is_set():
                clear_output()
                sleep(0.1)
                print(self,end="")   
                self._big_update_event.clear()
            else :
                print(self._get_epoch_line(),end="")
            self._update_event.clear()
            sleep(2)
    #endregion
    
    #region __str__ methods
    # Print current state of the Trainer     
    def _to_str(self,info:Any) -> str:
        if isinstance(info,float):
            return f"{info:.2f}"
        return str(info) 
    
    def _line(self,call: Callable[[Any],Any]) -> str:
        infos = [self._to_str(call(i)) for i in self.threads]
        return " | ".join([j + " "*(self._line_size-len(j)) for j in infos])
    def _line2(self,call: Callable[[Any],Tuple[Any,Any]]) -> str:
        infos = [call(i) for i in self.threads]
        infos = [(self._to_str(j1),self._to_str(j2)) for j1,j2 in infos]
        return " | ".join([j1 + " "*(self._line_size-len(j1)-len(j2)) +j2 for j1,j2 in infos])
    def _line_center(self,call: Callable[[Any],Any]) :
        infos = [self._to_str(call(i)) for i in self.threads]
        return " | ".join([i.center(self._line_size) for i in infos])
    
    def _get_thread_info(self,thread,info_name)->str:
        t: ThreadedTrainer.Trainers = thread
        if info_name == "epoch":
            return self._to_str(t.epoch)
        if info_name in ["epoch","size","device","accuracy","last_update"]:
            return self._to_str(t.info[info_name])
        if t.network is None:
            return "NaN"
        if info_name == "model":
            return t.network.model.name
        if info_name == "dataset":
            return t.network.dataset.name
        if info_name == "optimizer":
            return t.network.optimizer.name
        if info_name == "max_epoch":
            return self._to_str(t.network.epoch)
        return "NaN"
    def _get_epoch(self,thread)->str:
        t: ThreadedTrainer.Trainers = thread
        return t.info["epoch"]
    
    def _get_warning(self):
        if len(self.warning) == 0:
            return ""
        return "WARNING :\n\t" + "\n\t".join(self.warning) + "\n\n\n"
    def _get_task_remaining(self) -> str:
        a = "TASKS REMAINING :\n"
        
        l = [str(i+1)+". " + x[0] for i,x in enumerate(self.queue)]
        ml = max([len(i) for i in l],default=0)
        for i in range(len(l)//2):
            a += "    " + l[i] 
            a += " "*(ml-len(l[i])+10)
            a +=  l[len(l)//2 + i + len(l)%2]+"\n"
        if len(l)%2 == 1:
            a += "    " + l[len(l)//2] + "\n"             
        return a + "\n"
    def _get_general_info(self) -> str:
        a = "GENERAL INFO :\n"
        a += "    " + f"Accuracy updated every {self.saving_step} epochs\n"
        if torch.cuda.is_available():
            available,total = torch.cuda.mem_get_info()
            max_usage = torch.cuda.max_memory_allocated()
            a+= "    " + "GPU available: "+ self._to_str(available/1e9) + "GB out of " + self._to_str(total/1e9) + "GB\n"
            a+= "    " + "GPU max usage: "+ self._to_str(max_usage/1e9) + "GB out of " + self._to_str(total/1e9) + "GB\n"
        return a + "\n"
    def _get_result(self) -> str:
        a = "RESULTS :\n"
        a+= "    " + "     Model     |   Optimizer   |    Dataset    ||    optimization_parameters   ||    accuracy   \n"
        a+= "    " + "---------------|---------------|---------------||------------------------------||---------------\n"
        for key,item in self.results.items():
            data = key.split("_")
            model = data[0]
            opt = data[1]
            dataset = data[2]
            param = data[3] if len(data) == 4 else "¤"
            a += "    " +  f"{model.center(15)}|{opt.center(15)}|{dataset.center(15)}||"
            a += f"{param.center(30)}||"+self._to_str(item).center(15)+"\n"

        return a + "\n"        
    def _get_epoch_line(self)->str:    
        return "    " + "Epoch     || " + self._line(lambda x : self._get_thread_info(x,"epoch")) + " ||\r"       
    def _get_thread(self)->str:
        a = "THREADS :\n"
        a+= "    " + "Thread    || " + self._line_center(lambda x: x.name)                                      + " ||\n"
        a+= "    " + "----------||-" + "-|-".join(["-"*self._line_size for i in self.threads])                   + "-||\n"
        a+= "    " + "Model     || " + self._line(lambda x: self._get_thread_info(x,"model"))                   + " ||\n"
        a+= "    " + "Dataset   || " + self._line(lambda x: self._get_thread_info(x,"dataset"))                 + " ||\n"
        a+= "    " + "Optimizer || " + self._line(lambda x: self._get_thread_info(x,"optimizer"))               + " ||\n"
        a+= "    " + "cuda      || " + self._line(lambda x: self._get_thread_info(x,"device"))                  + " ||\n"
        a+= "    " + "epoch max || " + self._line(lambda x: self._get_thread_info(x,"max_epoch"))               + " ||\n"
        a+= "    " + "Size      || " + self._line(lambda x: self._get_thread_info(x,"size"))                    + " ||\n"
        a+= "    " + "----------||-" + "-|-".join(["-"*self._line_size for i in self.threads])                   + "-||\n"
        a+= "    " + "Accuracy  || " + self._line2(
            lambda x :(self._get_thread_info(x,"last_update") + ":" , self._get_thread_info(x,"accuracy")+"%")) + " ||\n"
        return a        
                
    def __str__(self):
        a = self._get_warning()
        a += self._get_result() 
        a += self._get_task_remaining()
        a += self._get_general_info()
        a += self._get_thread()
        a += self._get_epoch_line()
        return a    
    #endregion
    class Trainers(Thread):
        def __init__(self,parent,name:Union[str,None] = None) -> None:
            """A thread that will train a model

            Args:
                parent (ThreadedTrainer): The parent ThreadedTrainer
                name (Union[str,None], optional): Name of the threads. Defaults to None.
            """
            super().__init__(name=name)
            self.parent :ThreadedTrainer = parent
            #runtime info
            self.epoch = 0
            self.info :Dict[str,Any] = self._default_info()
            self.network :Union[Network,None] = None
            self._done = Event()
            #region CONSTANTS
            self.DATA_PATH = self.parent.DATA_PATH
            #endregion
            #init 
            self._set_info_default()
                    
        def _default_info(self) -> dict:
            return  {
            "accuracy":0,
            "device":"cpu",
            "size":0,
            "last_update":0,
            "key":"NaN" 
            }   
        
        def _set_info(self,device:str,size:int,key:str,Network:Network):
            """Set a info key, notify the parent"""
            self.info["device"] = device
            self.info["size"] = size
            self.info["key"] = key
            self.network = Network
            self.parent._big_update()
            
        def _set_info_default(self):
            """Set the info to default value, notify the parent, and set the thread as done, if all threads are done, notify the parent
            """
            self.info.update(self._default_info())
            self.network = None
            self.epoch = 0
            self.parent._big_update()
            self._done.set()
            for t in self.parent.threads:
                if not t._done.is_set():
                    return
            self.parent.done.set()

        def _train(self,key:str,net:Network,non_zero:bool) -> None:
            """Train a model

            Args:
                key (str): Name of the model
                net (Network): Description of the model
                non_zero (bool): If True, number of non zero parameters will be saved

            Returns:
            """            
            #region DEVICE
            DEVICE_NAME = "cuda" if net.use_cuda and torch.cuda.is_available() else "cpu"
            DEVICE = torch.device(DEVICE_NAME)
            #endregion
            
            #region DataSets
            class_size,trainset,testset = dset.get_dataset(net.dataset,self.DATA_PATH)
            dataLoader = lambda train,dataset : DataLoader(dataset, batch_size=net.batch_size, shuffle= train)
            trainLoader,testLoader =  dataLoader(True,trainset), dataLoader(False,testset)
            #endregion
            
            #region Model
            model :nn.Module = n_cl.get_model(net.model,trainset[0][0].shape[1],class_size).to(DEVICE)
            #endregion
            
            #region Training method
            criterion = nn.CrossEntropyLoss()
            optimizer = n_optim.get_optimizer(net.optimizer,model)       
            scheduler = n_optim.get_scheduler(net.scheduler,optimizer)
            #endregion
            
            #region Recover
            start = self.parent._get_network(key,model,optimizer)
            #endregion
            
            #region Training
            self._set_info(DEVICE_NAME,n_utils.count_parameters(model),key,net)
            iters = len(trainLoader)
            model.train()
            
            #region epoch loop
            if non_zero: self.parent._add_non_zero(key,model) 
            for self.epoch in range(start,net.epoch):
                self.parent._update()
                
                #region train    
                for i,(inputs,labels) in enumerate(trainLoader,0):
                    inputs,labels = inputs.to(DEVICE),labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(self.epoch+ i/iters)
                #endregion
                
                if non_zero: self.parent._add_non_zero(key,model)      
                
                if (self.epoch % self.parent.saving_step == self.parent.saving_step-1) or self.parent._kill.is_set():
                    self.info["accuracy"] = n_utils.get_accuracy(model, testLoader,DEVICE)
                    self.info["last_update"] = self.epoch+1
                    self.parent._big_update()
                    self.parent._save_incomplete(key,model,optimizer,self.epoch+1)
                    if non_zero: self.parent._save_non_zero(key)                 
                    if self.parent._kill.is_set():
                        return 
                
                if self.epoch == 0: self.parent._big_update() # To get max GPU usage update
            #endregion
            
            #endregion
            self.parent._save(key,model,n_utils.get_accuracy(model, testLoader,DEVICE))
            if non_zero: self.parent._save_non_zero(key)
                                  
        def run(self) -> None:
            """Run the thread, train a model, and save it, if the thread is killed, save the model and exit
            
            Will wait for queue to be not empty, then get the next model to train
            
            Raises:
                e: Cuda out of memory error the thread will retry 3 times with interval (15s,30s,60s). If the error is still there will pause till all other threads are done. 
                e: Any other error will be logged and the thread will be Terminated
            """
            error_time_out = 15
            error_counter = 0
            buffer:Union[Tuple[str,Network,bool],None] = None
            try :
                while( not self.parent._kill.is_set()):
                    for w in self.parent.warning:
                        if self.name in w and "CUDA out of memory" in w:
                            self.parent.warning.remove(w)
                    self.parent._available.wait()
                    buffer = self.parent._get_next_in_queue()
                    if buffer is None:
                        continue
                    try :
                        self._done.clear()
                        self._train(*buffer)
                        self._set_info_default()
                        error_counter = 0
                        sleep(5)
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            self._set_info_default()
                            if error_counter < 3:
                                self.parent._error(self.name,buffer,
                                                   f"CUDA out of memory, will retry in {error_time_out} seconds ({error_counter+1}/3)")
                                sleep(error_time_out)
                                error_time_out *= 2
                                error_counter +=1
                            else :
                                self.parent._error(self.name,buffer,
                                                   f"CUDA out of memory, will pause till every other trainer is done")
                                self.parent.done.wait()
                                error_counter = 0
                                error_time_out = 15
                            continue
                        else:
                            raise e
            except Exception as e:
                self._set_info_default()
                self.parent._error(self.name,buffer,e)
                raise e


def optimization_thread(optim:dict,network_name:str,trainer:ThreadedTrainer):
    Thread(target=network_Optimizer,args=(optim,network_name,trainer)).start()

def network_Optimizer(optim:dict,network_name:str,trainer:ThreadedTrainer):
    optim_dict = optim["optim"]
    optim["networks"][network_name] 

    to_optimize = []
    #get the parameters to optimize
    for structure in ["model","dataset","optimizer","scheduler"]:
        for param, value in optim_dict[structure][optim["networks"][network_name] [structure]["name"]].items():
            to_optimize.append((structure,param,value))
    #add result
    optim[network_name] = {"accuracy":0,"best":{}}
    for structure,param,_ in to_optimize:
        optim[network_name]["best"][structure+"."+param] = optim["networks"][network_name] [structure][param]
    #get baseline
    trainer.add(network_name+"_baseline",Network(optim["networks"][network_name]) )
    trainer.done.wait()
    optim[network_name]["accuracy"] = trainer.results[network_name+"_baseline"]
    #optimize
    for structure,param,value in to_optimize:
        for i in [val for val in to_optimize if val != optim["networks"][network_name] [structure][param]]:
            network_copy = deepcopy(optim["networks"][network_name] )
            network_copy[structure][param] = i
            trainer.add(network_name+"_"+structure+"."+param.replace("_","-")+"="+i,Network(network_copy))
        trainer.done.wait()
        for i in [val for val in to_optimize if val != optim["networks"][network_name][structure][param]]:
            new_name = network_name+"_"+structure+"."+param.replace("_","-")+"="+i
            if trainer.results[new_name] > optim[network_name]["accuracy"]:
                optim[network_name]["best"][structure+"."+param] = i
                optim[network_name]["accuracy"] = trainer.results[new_name]

    pass 
    