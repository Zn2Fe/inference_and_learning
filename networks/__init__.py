# Author: Nicolas DEVAUX

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from typing import Tuple,List,Callable,Union,Dict

import networks.utils as n_utils
import networks.dataset as dset
import networks.custom_layers as cl
import networks.optimizers as noptim
from threading import Thread,Lock
from time import sleep
import json
from IPython.display import clear_output
import os

 

class NetworkOptim:
    """Interface to optimize the hyperparameters of a network
    
    Args:
        optim (dict): dictionary containing the hyperparameters to optimize
        DATA_PATH (str): path to the loacation of the dataset
        epoch (int, optional): number of epoch to train the network. Defaults to 100.
        verbose (bool, optional): verbose mode. Defaults to False.
        very_verbose (bool, optional): very verbose mode. Defaults to False.
        
    """
    def __init__(self,optim:dict,DATA_PATH,epoch=100,verbose = False, very_verbose = False) -> None:
        self.optim = optim
        self.verbose = verbose
        self.very_verbose = very_verbose
        self.DATA_PATH = DATA_PATH
        self.epoch = epoch
    
    def __call__(self,net:dict):
        """Optimize the hyperparameters of a network
        
        Args:
            net (dict): dictionary containing the network to optimize
        """
        net["epoch"] = self.epoch
        if self.verbose: print("Calculating accuracy for default parameters")
        accuracy,_,_ = train(net,self.DATA_PATH,verbose=True,very_verbose=self.very_verbose)
        best = {}
        for to_optim in ["model","dataset","optimizer","scheduler"]:
            for key,item in self.optim[to_optim][net[to_optim]["name"]].items():
                if self.verbose:print(f"Optimizing {to_optim}.{key} with {item}")
                best[to_optim+"."+key] = net[to_optim][key]
                acc,item = self.__train_list(net,to_optim,key,[i for i in item if i!= net[to_optim][key]])
                if acc > accuracy:
                    accuracy = acc
                    best[to_optim+"."+key] = item
                net[to_optim][key] = best[to_optim+"."+key]
                if self.verbose:print(f"Best accuracy for {to_optim}.{key} : {accuracy} with {best[to_optim+'.'+key]}")
        return accuracy,best
                
    def __train_list(self,net,key1,key2,items):
        accuracy = 0
        best = items[0]
        for item in items:
            net[key1][key2] = item
            if self.verbose: print(f"Calculating accuracy for {key1}.{key2} = {item}")
            acc,_,_ = train(net,self.DATA_PATH,verbose=self.verbose,very_verbose=self.very_verbose)
            if acc > accuracy:
                accuracy = acc
                best = item
            if self.verbose: print(f"Accuracy for {key1}.{key2} = {item} : {acc}")
        return accuracy,best
class OptimThread(Thread):
    def __init__(self,optim : dict ,items :Tuple[str,dict], net_optim : NetworkOptim):
        Thread.__init__(self)
        self.optim = optim
        self.key,self.item = items
        self.net_optim = net_optim
    
    def run(self):
        print(f"Starting to run thread for {self.key}")
        accuracy,best = self.net_optim(self.item)
        n_utils.dict_printer(best)
        print("Accurary :",accuracy )   
        self.optim["results"][self.key] = {}
        self.optim["results"][self.key]["best"] = best
        self.optim["results"][self.key]["accuracy"] = accuracy
        print(f"Thread for {self.key} finished")    
# implementation according to pytorch documentation
def train(net:dict, DATA_PATH,save_number_of_non_zero = False,verbose=False,very_verbose=False,line_size = 100):
    """Train a network
    
        Args:
            net (dict): dictionary containing the network to train
            DATA_PATH (str): path to the location of the dataset
            save_number_of_non_zero (bool, optional): save the number of non zero parameters for S_[CONV,LOCAL,FC]. Defaults to False.
            verbose (bool, optional): verbose mode. Defaults to False.
        
        Returns:
            accuracy (float): accuracy of the network
            model (nn.Module): trained model
            number_of_non_zero (dict): number of non zero parameters for S_[CONV,LOCAL,FC]
    """ 
    #CUDA
    use_cuda = net["use_cuda"]
    if use_cuda and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        use_cuda = False
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    
    #DataSets
    class_size,trainset,testset = dset.get_dataset(net["dataset"],DATA_PATH)
    dataLoader = lambda train,dataset : DataLoader(dataset, batch_size=net["batch_size"], shuffle= train, num_workers=2)
    trainLoader,testLoader =  dataLoader(True,trainset), dataLoader(False,testset)
    
    #Model
    model :nn.Module = cl.get_model(net["model"],trainset[0][0].shape[1],class_size).to(DEVICE)
    number_of_non_zero = {}
    if save_number_of_non_zero:
        if not isinstance(model,cl.S_Conv):
            raise ValueError("Model must be S_Conv")
        number_of_non_zero = {
            "conv_like":[n_utils.count_non_zero_parameters(model,"conv_like")],
            "FC1":[n_utils.count_non_zero_parameters(model,"FC.0")],
            "FC2":[n_utils.count_non_zero_parameters(model,"FC.3")],
        }
    #Training method
    criterion = nn.CrossEntropyLoss()
    optimizer = noptim.get_optimizer(net["optimizer"],model)
    scheduler = noptim.get_scheduler(net["scheduler"],optimizer)
    
    if verbose: print(f"Training {n_utils.pd_dict_to_string(net,model)} on {DEVICE}")
    iters = len(trainLoader)
    for epoch in range(net["epoch"]):
        for i,(inputs,labels) in enumerate(trainLoader,0):
            inputs,labels = inputs.to(DEVICE),labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch+ i/iters)
        if verbose:
            if very_verbose : print("%.2f" % n_utils.get_accuracy(model, testLoader,DEVICE),end='%, ')
            else : print("¤",end='')
            if(epoch % line_size == line_size-1) : print(f' : {epoch+1}/{net["epoch"]}')
            elif(epoch == net["epoch"]-1): print(f' : {epoch+1}/{net["epoch"]}')
        if save_number_of_non_zero:
            number_of_non_zero["conv_like"].append(n_utils.count_non_zero_parameters(model,"conv_like"))
            number_of_non_zero["FC1"].append(n_utils.count_non_zero_parameters(model,"FC.0"))
            number_of_non_zero["FC2"].append(n_utils.count_non_zero_parameters(model,"FC.3"))
    
    acc = n_utils.get_accuracy(model, testLoader,DEVICE)
    if verbose:
        print("Finish training !")
        print(f"Accuracy : {acc}%")
    return acc,model,number_of_non_zero


class ThreadedTrainer(Thread):  
    def __init__(self,
                 num_threads:int = 1,
                 DATA_PATH:str = ".",
                 SAVE_TABLE_PATH:Union[str,None]=None,
                 SAVE_NETWORK_PATH:str="./networks_saved/",
                 SAVE_FIGURE3_PATH:str="./files/figure3.json",
                 mode="trainer",
                 line_size = 11,acc_update=100) -> None:
        super().__init__(name="ThreadedTrainer")
        self.logger: Logger
        self._lock = Lock()
        self.threads = []
        self.queue = []
        self.results = {}
        self.mode = mode
        self.num_threads = num_threads
        self.save_path = SAVE_TABLE_PATH
        
        
        self.new_thread = lambda name : ThreadedTrainer.Trainers(
            self._get_next_in_queue,
            self._add_results,
            self._error,
            self._is_alive,
            DATA_PATH,
            SAVE_NETWORK_PATH,
            SAVE_FIGURE3_PATH,
            acc_update,
            name=name)
        
        self.acc_update = acc_update      
        self.line_size = line_size
    
    def add(self,key,net,non_zero=False):
        with self._lock:
            self.queue.append((key,net,non_zero))
        
    def _get_next_in_queue(self) -> Union[Tuple[str,dict,bool], None]:
        with self._lock:
            if len(self.queue):
                return self.queue.pop(0)
            return None
        
    def _add_results(self,key,res):
        with self._lock:
            self.results[key] = res
    
    def _error(self,thread_name,buff,error):
        self.add(*buff)
        self.logger.warning.append(f"{thread_name} is disabled, du to {error}")
    
    def _is_alive(self):
        if self.mode == "trainer":
            return len(self.threads) > 0
        elif self.mode == "optimizer":
            return self.mode == "optimizer"
        else : return self.mode != "stop"
    
    def stop(self):
        self.mode = "stop"
    class Trainers(Thread):
        def __init__(self,
                     next : Callable[...,Union[Tuple[str,dict,bool], None]],
                     res:Callable[[str,float],None],
                     error:Callable[[str,Union[Tuple[str,dict,bool], None],str],None],
                     is_alive:Callable[[],bool],
                     DATA_PATH:str,
                     SAVE_NETWORK_PATH:str,
                     SAVE_FIGURE3_PATH:str,
                     acc_update,
                     name:Union[str,None] = None) -> None:
            super().__init__(name=name)
            #callable
            self.next = next
            self.res = res
            self.error = error
            self.is_alive = is_alive
            
            #runtime info
            self.epoch = 0
            self.notifier = False
            self.info = {"acc_update":acc_update}
            self.set_info()
            #constants
            self.DATA_PATH = DATA_PATH
            self.SAVE_NETWORK_PATH = SAVE_NETWORK_PATH
            self.SAVE_FIGURE3_PATH = SAVE_FIGURE3_PATH
                 
        def set_info(self):
            self.info.update({
            "model":"NaN",
            "optimizer":"NaN",
            "dataset":"NaN",
            "accuracy":0,
            "device":"cpu",
            "max_epoch":0,
            "size":0
            })

            self.notifier = True
                                        
        def run(self) -> None:
            error_flag = False
            empty_flag = False
            while( self.is_alive()):
                buffer = self.next()
                if buffer is not None:
                    self.set_info()
                    try :
                        
                        res = self.train(*buffer)
                        self.res(buffer[0],res)
                        buffer = self.next() 
                        sleep(5)
                        error_flag = False
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e) and not error_flag:
                            error_flag = True
                            sleep(60)
                            continue
                        elif "CUDA out of memory" in str(e) and error_flag:
                            self.error(self.name,buffer,str(e))
                            buffer = None
                            break
                        else:
                            raise e
                else :
                    if not empty_flag:
                        self.set_info()
                    empty_flag = True
                    sleep(5)
        
        def train(self,key,net,non_zero) -> float:
            self.info["device"] = "cuda" if net["use_cuda"] and torch.cuda.is_available() else "cpu"
            DEVICE = torch.device(self.info["device"])
            
            #region DataSets
            class_size,trainset,testset = dset.get_dataset(net["dataset"],self.DATA_PATH)
            dataLoader = lambda train,dataset : DataLoader(dataset, batch_size=net["batch_size"], shuffle= train, num_workers=2)
            trainLoader,testLoader =  dataLoader(True,trainset), dataLoader(False,testset)
            self.info["dataset"] = net["dataset"]["name"]
            #endregion
            
            #region Model
            model :nn.Module
            if os.path.isfile(self.SAVE_NETWORK_PATH+f"{key}_INCOMPLETE_{self.epoch}.pth"):
                model = torch.load(self.SAVE_NETWORK_PATH+f"{key}_INCOMPLETE_{self.epoch}.pth")
            else :
                model = cl.get_model(net["model"],trainset[0][0].shape[1],class_size).to(DEVICE)
            number_of_non_zero = {}

            self.info["size"] = str(n_utils.count_parameters(model)/1e6)[:6] + " M" 
            self.info["model"] = net["model"]["name"]
            
            if non_zero:
                if not isinstance(model,cl.S_Conv):
                    raise ValueError("Model must be S_Conv")
                number_of_non_zero = {
                    "conv_like":[n_utils.count_non_zero_parameters(model,"conv_like")],
                    "FC1":[n_utils.count_non_zero_parameters(model,"FC.0")],
                    "FC2":[n_utils.count_non_zero_parameters(model,"FC.3")],
                }
            #endregion
            
            #region Training method
            criterion = nn.CrossEntropyLoss()
            optimizer:torch.optim.Optimizer
            if os.path.isfile(self.SAVE_NETWORK_PATH+f"{key}_INCOMPLETE_{self.epoch}.pth"):
                optimizer = torch.load(self.SAVE_NETWORK_PATH+f"{key}_OPTIM_INCOMPLETE_{self.epoch}.pth")
            else :
                optimizer = noptim.get_optimizer(net["optimizer"],model)
                
            scheduler = noptim.get_scheduler(net["scheduler"],optimizer)
            self.info["optimizer"] = net["optimizer"]["name"]
    
            #endregion
            
            #region Training
            iters = len(trainLoader)
            self.info["max_epoch"] = net["epoch"]
            self.notifier = True
            #Start training
            for self.epoch in range(net["epoch"]):
                for i,(inputs,labels) in enumerate(trainLoader,0):
                    inputs,labels = inputs.to(DEVICE),labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(self.epoch+ i/iters)
                
                if non_zero:
                    number_of_non_zero["conv_like"].append(n_utils.count_non_zero_parameters(model,"conv_like"))
                    number_of_non_zero["FC1"].append(n_utils.count_non_zero_parameters(model,"FC.0"))
                    number_of_non_zero["FC2"].append(n_utils.count_non_zero_parameters(model,"FC.3"))
                
                if (self.epoch % self.info["acc_update"] == self.info["acc_update"]-1):
                    self.info["accuracy"] = n_utils.get_accuracy(model, testLoader,DEVICE)
                    torch.save(model.state_dict(),self.SAVE_NETWORK_PATH+f"{key}_INCOMPLETE_{self.epoch}.pth")
                    torch.save(optimizer.state_dict(),self.SAVE_NETWORK_PATH+f"{key}_OPTIM_INCOMPLETE_{self.epoch}.pth")
             
            acc = n_utils.get_accuracy(model, testLoader,DEVICE)
            torch.save(model.state_dict(),self.SAVE_NETWORK_PATH+f"{key}.pth")
            os.remove(self.SAVE_NETWORK_PATH+f"{key}_INCOMPLETE_{self.epoch}.pth")
            os.remove(self.SAVE_NETWORK_PATH+f"{key}_OPTIM_INCOMPLETE_{self.epoch}.pth")
            if non_zero:
                json.dump(number_of_non_zero,open(self.SAVE_FIGURE3_PATH,"w"))
            #endregion
            
            return acc

        
    def run(self):
        for i in range(self.num_threads):
            self.threads.append(self.new_thread("Trainer"+str(i)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for t in self.threads:
            t.start()
        self.logger = Logger(self,self.line_size)
        for t in self.threads:
            t.join()
        self.logger.stop()
        if self.save_path is not None:
            table2 = pd.read_csv(self.save_path)
            for key,item in self.results.items() :
                table2.loc[table2["Model"] == key.split("_")[0] and table2["Training_Method"] == key.split("_")[1] ,key.split("_")[2]] = item
            table2.to_csv(self.save_path,index=False) 
    

class Logger(Thread):
        def __init__(self,trainer:ThreadedTrainer,line_size=11):
            Thread.__init__(self)
            self.line_size = line_size
            self.trainer:ThreadedTrainer = trainer
            self.threads : List[ThreadedTrainer.Trainers] = trainer.threads
            self._continue = True
            
            self.getter = lambda call : " | ".join([j + " "*(self.line_size-len(j)) for j in [str(call(i)) for i in self.threads]])
            self.getter_double = lambda call : " | ".join([j1 + " "*(self.line_size-len(j1)-len(j2)) +j2 for j1,j2 in [call(i) for i in self.threads]])
            self.getter_center =  lambda call : " | ".join([str(call(i)).center(self.line_size) for i in self.threads])
            self.warning = []
            self.start()
            
        def get_warning(self):
            if len(self.warning) == 0:
                return ""
            return "WARNING !!! :\n" + "\n".join(self.warning) + "\n\n\n"

        def get_result(self) -> str:
            if self.trainer.mode == "trainer":
                a = "     Model     |   Optimizer   |    Dataset    ||    accuracy    \n"
                a+= "---------------|---------------|---------------||----------------\n"
                for key,item in self.trainer.results.items():
                    model,opt,dataset = key.split("_")
                    a += f"{model.center(15)}|{opt.center(15)}|{dataset.center(15)}||"+"{0:.2f}%".format(item).center(15)+"\n"
                a+= "\n\n"
                return a 
            if self.trainer.mode == "optimizer":
                a = "     Model     |   Optimizer   |    Dataset    ||    optimization_parameters   ||    accuracy   \n"
                a+= "---------------|---------------|---------------||------------------------------||---------------\n"
                for key,item in self.trainer.results.items():
                    model,opt,dataset,param = key.split("_")
                    a += f"{model.center(15)}|{opt.center(15)}|{dataset.center(15)}||"
                    a += f"{param.center(30)}||"+"{0:.2f}%".format(item).center(15)+"\n"
                a+= "\n\n"
                return a
            return ""   
        
        def get_epoch_line(self)->str:
            return "epoch,acc || " + self.getter_double(lambda x :( str(x.epoch),"{0:.2f}%".format(x.info["accuracy"])) ) + " ||"
                 
        def __str__(self):
            a = self.get_warning()
            a += f"{len(self.trainer.queue)} tasks remaining)\n" 
            a += f"Accuracy updated every {self.trainer.acc_update} epochs\n"
            if torch.cuda.is_available():
                available,total = torch.cuda.mem_get_info()
                max_usage = torch.cuda.max_memory_allocated()
                a+= "GPU available: "+ str(available/1e9) + "GB out of " + str(total/1e9) + "GB\n"
                a+= "GPU max usage: "+ str(max_usage/1e9) + "GB out of " + str(total/1e9) + "GB\n"
            a+= "\n\n"
            
            a+= self.get_result() 
            
            a+= "Thread    || " + self.getter_center(lambda x: x.name)                   + " ||\n"
            a+= "----------||-" + "-|-".join(["-"*self.line_size for i in self.threads]) + "-||\n"
            a+= "Model     || " + self.getter(lambda x: x.info["model"])                 + " ||\n"
            a+= "Dataset   || " + self.getter(lambda x: x.info["dataset"])               + " ||\n"
            a+= "Optimizer || " + self.getter(lambda x: x.info["optimizer"])             + " ||\n"
            a+= "cuda      || " + self.getter(lambda x: x.info["device"])                + " ||\n"
            a+= "epoch max || " + self.getter(lambda x: x.info["max_epoch"])             + " ||\n"
            a+= "Size      || " + self.getter(lambda x: x.info["size"])                  + " ||\n"
            a+= "----------||-" + "-|-".join(["-"*self.line_size for i in self.threads]) + "-||\n"
            a+= self.get_epoch_line()
            return a
        
        def run(self):
            while self._continue:
                if any([i.notifier for i in self.threads]):
                    for i in self.threads:
                        i.notifier = False
                    clear_output(wait=True)
                    sleep(0.1)
                    print(self,end="\r")
                else :
                    print(self.get_epoch_line(),end="\r")
                sleep(0.1)
            clear_output(wait=False)
            sleep(0.1)
            print(self.get_result(),end="\n")
                    
        def stop(self):
            self._continue = False

