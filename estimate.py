from transformers import  BertTokenizer
import torch
import argparse
from modeling_cpt import CPTForConditionalGeneration
from utils import get_pred
from rouge import Rouge

def process_line(line):
    length = len(line)
    ret = ['']
    for i in range(length-1):
        ret[0] += line[i]
        ret[0] += ' '
    ret[0] += line[length-1]
    return ret

def process_rouge(score):
    length = len(score)
    sum_r = 0
    sum_p = 0
    sum_f = 0
    for i in range(length):
        sum_r += score[i]['r']
        sum_p += score[i]['p']
        sum_f += score[i]['f']
    ret = ({'r':sum_r/length},{'p':sum_p/length},{'f':sum_f/length})
    return ret

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
    lines = ""
    input_doc = ""
    # import pdb;pdb.set_trace()
    flag = 0
    input = ''
    judger = Rouge()
    Rouge_1 = []
    Rouge_2 = []
    Rouge_l = []
    for line in open(arg_dict['file_path'],'r',encoding="UTF-8"):  # 会以\n划分，每个line包含末尾的'\n'
        lines += line
        # print(line)
        if flag == 0:
            flag = 1
            input = line
        else:
            flag = 0
            result = line
            # import pdb;pdb.set_trace()
            result = process_line(result)
            output = get_pred(tokenizer, model, input, sum_min_len=int(arg_dict['sum_min_len']), device=device)
            # print(output)
            score = judger.get_scores(output,result)
            Rouge_1.append(score[0]["rouge-1"])
            Rouge_2.append(score[0]["rouge-2"])
            Rouge_l.append(score[0]["rouge-l"])
    score_1 = process_rouge(Rouge_1)
    score_2 = process_rouge(Rouge_2)
    score_l = process_rouge(Rouge_l)
    print('Rouge_1 : ',score_1)
    print('Rouge_2 : ',score_2)
    print('Rouge_l : ',score_l)
