import pickle


from tqdm import tqdm

import random
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset


from utils import init_logger


logger = init_logger()

# Define the input Example class
class MTRSExample(object):
    '''Unprocessed sample class'''
    def __init__(self,guid, utterences, response, label):
        self.guid=guid
        self.utterences = utterences
        self.response = response
        self.label = label


# 定义输入feature类
class MTRSFeatures(object):
    """Samples processed as input features"""

    def __init__(self,
                 utters_id,
                 utters_len,
                 utters_num,
                 response_id,
                 response_len,
                 label=None):
        self.utters_id = utters_id
        self.utters_len = utters_len
        self.utters_num = utters_num
        self.response_id = response_id
        self.response_len = response_len
        self.label = label


# Define the expected processor of the task
class UbuntuCorpus(object):
    def __init__(self,args):
        super(UbuntuCorpus, self).__init__()
        self.args=args
        #self.embed_file=os.path.join(self.args.data_dir,'embedding.pkl')

        self.train_file = os.path.join(self.args.data_dir,'utterances.pkl')
        self.eval_file=os.path.join(self.args.data_dir,'Evaluate.pkl')

        self.output_dir=self.args.output_dir
        self.data_file=os.path.join(self.args.output_dir,'data.pt')

        self.load()

    def load(self):
        # Build if not available
        if not os.path.exists(self.data_file):
            logger.info("Build Corpus ...")
            self.build()
        else:
            self.data=torch.load(self.data_file)
        # Load if available

    def build(self):
        logger.info("Reading Data ...")
        train_examples = self.read_and_build_examples( data_type="train")
        eval_examples = self.read_and_build_examples(data_type="eval")
        self.data = {"train": train_examples,
                     "eval": eval_examples
                     }

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        torch.save(self.data, self.data_file)

    def read_and_build_examples(self, data_type='train'):
        examples = []
        data_to_convert = None
        if data_type == "train":
            data_to_convert=pickle.load(open(self.train_file,'rb'))
        elif data_type == "eval":
            data_to_convert=pickle.load(open(self.eval_file,'rb'))
        n_sample=len(data_to_convert[0])
        for i in range(n_sample):
            utterance=data_to_convert[0][i]
            response=data_to_convert[1][i]
            label=data_to_convert[2][i]
            #print(f"\t[\n\t\t{utterance},\n\t\t{label},\n\t\t{response}\n\t],")
            examples.append(MTRSExample(guid=i,
                utterences=utterance,
                response=response,
                label=label))
        return examples
        """
        Read the original data and construct the sample object MTRSExample
        """
        examples=[]
        #Training set sample generation——self.data['train']
        if data_type == "train":
            data_to_convert=pickle.load(open(self.train_file,'rb'))
            n_sample=len(data_to_convert[0])
            # Random order (not strictly pseudo-random, but good enough)
            rand_bias=random.randint(0,n_sample)
            neg_response_ids=[(i+rand_bias)%n_sample for i in range(n_sample)]

            for i in range(n_sample):
                utterance=data_to_convert[0][i]
                pos_response=data_to_convert[1][i]
                neg_response= data_to_convert[1][neg_response_ids[i]]
                #print(f"\t[\n\t\t{utterance},\n\t\t{label},\n\t\t{response}\n\t],")

                examples.append(MTRSExample(guid=i,
                    utterences=utterance,
                    response=pos_response,
                    label=1))
                examples.append(MTRSExample(guid=i,
                    utterences=utterance,
                    response=neg_response,
                    label=0))
        #Test set sample generation——self.data['eval']
        if data_type == "eval":
            data_to_convert=pickle.load(open(self.eval_file,'rb'))
            n_sample=len(data_to_convert[0])
            for i in range(n_sample):
                utterance=data_to_convert[0][i]
                response=data_to_convert[1][i]
                label=data_to_convert[2][i]
                #print(f"\t[\n\t\t{utterance},\n\t\t{label},\n\t\t{response}\n\t],")
                examples.append(MTRSExample(guid=i,
                    utterences=utterance,
                    response=response,
                    label=label))

        return examples

    def create_batch(self, data_type="train"):
        examples = self.data[data_type]
        features_cache_path = os.path.join(
            self.output_dir,
            "features-{}-{}-{}.pt".format(data_type, self.args.max_seq_length, self.args.max_utter_num)
        )
        if os.path.exists(features_cache_path):
            logger.info("Loading features from {} ...".format(features_cache_path))
            features = torch.load(features_cache_path)
        else:
            logger.info("Convert {} examples to features".format(data_type))
            features = self.convert_examples_to_features(examples, data_type)
            torch.save(features, features_cache_path)

        # ----------Modify as needed code here----------#
        all_utters_id = torch.tensor([f.utters_id for f in features], dtype=torch.long)
        all_utters_len = torch.tensor([f.utters_len for f in features], dtype=torch.long)
        all_utters_num = torch.tensor([f.utters_num for f in features], dtype=torch.long)
        all_response_id = torch.tensor([f.response_id for f in features], dtype=torch.long)
        all_response_len = torch.tensor([f.response_len for f in features], dtype=torch.long)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_utters_id, all_utters_len, all_utters_num, all_response_id, all_response_len, all_label)

        if data_type == "train":
            train_sampler = RandomSampler(dataset) if self.args.local_rank == -1 else DistributedSampler(dataset)
            dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        else:
            eval_sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
        #----------------------------------------#
        return dataloader



    def convert_examples_to_features(self, examples, data_type="train"):
        features = []
        desc_message = "GET Feature FROM " + data_type.upper()
        # self.embedding=pickle.load(open(self.embed_file,'rb'),encoding='bytes')
        for i,example in tqdm(enumerate(examples), desc=desc_message):
            #utterance filling, truncation
            utterances = example.utterences[-self.args.max_utter_num:]
            us_vec, us_len = [], []
            for utterance in utterances:
                if len(utterance)<=self.args.max_seq_length:
                    u_len=len(utterance)
                    u_vec=utterance+[0]*(self.args.max_seq_length-len(utterance))
                else:
                    u_len=self.args.max_seq_length
                    u_vec = utterance[:self.args.max_seq_length]
                us_vec.append(u_vec)
                us_len.append(u_len)
            if len(utterances)<self.args.max_utter_num:
                us_len+=[0]*(self.args.max_utter_num-len(utterances))
                us_num=len(utterances)
                us_vec+=(self.args.max_utter_num-len(utterances))*[[0]*self.args.max_seq_length]
            else:
                us_num=self.args.max_utter_num
            #response padding, truncation
            response=example.response
            if len(response)<=self.args.max_seq_length:
                r_len=len(response)
                r_vec=response+[0]*(self.args.max_seq_length-len(response))
            else:
                r_len=self.args.max_seq_length
                r_vec = response[:self.args.max_seq_length]
            features.append(MTRSFeatures(
                utters_id=us_vec,
                utters_len=us_len,
                utters_num=us_num,
                response_id=r_vec,
                response_len=r_len,
                label=example.label
            ))

        return features


#Test Cases
if __name__ == "__main__":
    from ModelConfig import SMNModel
    import torch.nn as nn
    loss_func=nn.CrossEntropyLoss()
    model=SMNModel()
    model.to('cuda:0')
    dataset=UbuntuCorpus()
    train_dataloader=dataset.create_batch('train')
    eval_dataloader=dataset.create_batch('eval')
    for i,batch in enumerate(eval_dataloader):
        utterance=batch[0].to('cuda:0')
        len_utterance=batch[1].to('cuda:0')
        num_utterance=batch[2].to('cuda:0')
        response=batch[3].to('cuda:0')
        len_response=batch[4].to('cuda:0')
        proper=batch[5].to('cuda:0')
        print('batch:',i)
        pred=model(utterance,len_utterance,num_utterance,response,len_response)
        loss = loss_func(pred,proper)

        print(loss)
        # if i==0:break
