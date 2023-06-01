from transformers import  BertTokenizer
import torch
import argparse
from modeling_cpt import CPTForConditionalGeneration
from utils import get_pred

from rouge import Rouge
from utils import get_loss

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",default='/path/to/model',type=str)
    parser.add_argument("--file_path",default="/path/to/file/",type=str)
    parser.add_argument("--sum_min_len",default=20,type=int)
    args = parser.parse_args()
    arg_dict=args.__dict__
    tokenizer = BertTokenizer.from_pretrained(arg_dict['model_path'])
    model = CPTForConditionalGeneration.from_pretrained(arg_dict['model_path']).to(device)
    lines=""
    input_doc=""
    for line in open(arg_dict['file_path'],'r',encoding="UTF-8"):
        lines+=line
    for sent in lines.split("\n"):
        input_doc+=sent

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(arg_dict['num_epoch']):
        loss = get_loss(tokenizer,model,input_doc,sum_min_len=int(arg_dict['sum_min_len']),device=device)
        optimizer.zero_grad()
        loss.backward()
    s=get_pred(tokenizer,model,input_doc,sum_min_len=int(arg_dict['sum_min_len']),device=device)
    print(s)
    target = arg_dict['target']
    judger = Rouge()
    score = judger.get_scores(s,target)
    print(score[0]["rouge-1"])
    print(score[0]["rouge-2"])
    print(score[0]["rouge-l"])
    PATH = 'state_dict_model.pth'
    torch.save(model.state_dict(),PATH)
