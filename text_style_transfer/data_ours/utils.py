import re

def check_quote_pair(paragraph):
    left_num = len(re.findall(u'“', paragraph, flags=re.U)) 
    right_num = len(re.findall(u'”', paragraph, flags=re.U))
    g_num = len(re.findall(u'"', paragraph, flags=re.U))
    left_suc_num = len(re.findall(u'“[^”“]*“', paragraph, flags=re.U))
    right_suc_num = len(re.findall(u'”[^”“]*”', paragraph, flags=re.U))
    start_right_num = len(re.findall(u'^[^”“]*”', paragraph, flags=re.U))
    end_left_num = len(re.findall(u'“[^”“]*$', paragraph, flags=re.U))
    # import pdb;pdb.set_trace()
    if left_suc_num> 0 or right_suc_num>0 or start_right_num >0 or end_left_num>0:
        return False
    if g_num>0:
        if g_num%2!=0 :
            return False
    if left_num >0 or right_num >0:
        if left_num!=right_num:
            return False
        else:
            return True
    else:
        return True
# def check_quote_pair_and_split(paragraph):
#     left_num = len(re.findall(u'“', paragraph, flags=re.U)) 
#     right_num = len(re.findall(u'”', paragraph, flags=re.U))
#     g_num = len(re.findall(u'"', paragraph, flags=re.U))
#     import pdb;pdb.set_trace()
#     if g_num>0:
#         if g_num%2!=0:
#             return ['']
#     if left_num >0 or right_num >0:
#         if left_num!=right_num:
#             temp = re.split(\
#                u'[^.?!……。？！"”]*“[^”“]*“[^”“]”[^.?!……。？！“”]*[.?!……。？！]+\
#                 |[^.?!……。？！]*“[^”“]*$\
#                 |[^.?!……。？！]*“[^”“]*”[^”“]*”[^.?!……。？！]*[^.?!……。？！]+', paragraph, flags=re.U)
#             import pdb;pdb.set_trace()
            
    

def sentence_split_zh(paragraph):
    import re
    if not check_quote_pair(paragraph):
        return ['']
    quotes_rep ='GGGHHHKKK'
    split_rep = 'SSSSS'
    #return re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]+', paragraph, flags=re.U)
    pure_quotes = re.findall(u'"[^",.?!……。，？！]+"|“[^“”,.?!……。，？！]+”', paragraph, flags=re.U)
    quotes = re.findall(u'"[^"]+"|“[^“”]+”', paragraph, flags=re.U)
    rep_list = [(x, quotes_rep+str(i)+'。') for i,x in enumerate(quotes) if x not in pure_quotes]
    paragraph_0 = paragraph
    for x,y in rep_list:
        paragraph_0=paragraph_0.replace(x,y)
    # if '六年后再写《面包树出走了》，写的也是我自己的成长和转变。' in paragraph:
    #     import pdb;pdb.set_trace()
    paragraph_1 = split_rep.join(re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]+', paragraph_0, flags=re.U))
    rep_list.reverse()
    for x,y in rep_list:
        paragraph_1 = paragraph_1.replace(y,x)

    r = paragraph_1.split(split_rep)
    
    for x in r:
        if not  check_quote_pair(x):
            return ['']
    
    return r

def sentence_split_indpdtquotes_zh(paragraph):
    import re
    if not check_quote_pair(paragraph):
        return ['']
    quotes_rep ='GGGHHHKKK'
    split_rep = 'SSSSS'
    #return re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]+', paragraph, flags=re.U)
    pure_quotes = re.findall(u'"[^",.?!……。，？！]+"|“[^“”,.?!……。，？！]+”', paragraph, flags=re.U)
    quotes = re.findall(u'"[^"]+"|“[^“”]+”', paragraph, flags=re.U)
    rep_list = [(x, quotes_rep+str(i)+'。') for i,x in enumerate(quotes) if x not in pure_quotes]
    if len(rep_list)>0:
        import pdb;pdb.set_trace()
    paragraph_0 = paragraph
    for x,y in rep_list:
        paragraph_0=paragraph_0.replace(x,'$'+y)
    # if '六年后再写《面包树出走了》，写的也是我自己的成长和转变。' in paragraph:
    #     import pdb;pdb.set_trace()
    paragraph_1 = split_rep.join(re.findall(u'[^!?。\.\!\?$]+[$!?。\.\!\?]+', paragraph_0, flags=re.U))
    rep_list.reverse()
    for x,y in rep_list:
        paragraph_1 = paragraph_1.replace(y,x)

    r = paragraph_1.split(split_rep)
    
    for x in r:
        if not  check_quote_pair(x):
            return ['']
    
    return [x.replace('$','') for x in r]


def clause_split_zh(paragraph):
    import re
    if not check_quote_pair(paragraph):
        return ['']
    quotes_rep ='GGGHHHKKK'
    split_rep = 'SSSSS'
    #return re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]+', paragraph, flags=re.U)
    pure_quotes = re.findall(u'"[^",.?!……。，？！]+"|“[^“”,.?!……。，？！]+”', paragraph, flags=re.U)
    quotes = re.findall(u'"[^"]+"|“[^“”]+”', paragraph, flags=re.U)
    rep_list = [(x, '。'+quotes_rep+str(i)+'。') for i,x in enumerate(quotes) if x not in pure_quotes]
    paragraph_0 = paragraph
    for x,y in rep_list:
        paragraph_0=paragraph_0.replace(x,y)
    # if '六年后再写《面包树出走了》，写的也是我自己的成长和转变。' in paragraph:
    #     import pdb;pdb.set_trace()
    paragraph_1 = split_rep.join(re.findall(u'[^!?。\.\!\?——:;：；]+[!?。\.\!\?——:;：；]+', paragraph_0, flags=re.U))
    rep_list.reverse()
    for x,y in rep_list:
        paragraph_1 = paragraph_1.replace(y,x)

    r = paragraph_1.split(split_rep)
    for i in range(len(r)):
        if r[i][-1] not in '.?!。？！……':
            r[i] = r[i].replace(r[i][-1],'。')
    for x in r:
        if not  check_quote_pair(x):
            return ['']
    
    return r



def en_len(x,tokenizer):
    return tokenizer.encode(x, return_tensors="pt").size(1)
def cut_string_en(input,max_len,tokenizer):
    return  cut_string(input,max_len,len_func = lambda x: en_len(x ,tokenizer))
def cut_string_zh(input,max_len):
    return  cut_string(input,max_len,split_func = sentence_split_zh,len_func = len)
def soft_cut_string_zh(input,max_len):
    return  soft_cut_string(input,max_len,split_func = sentence_split_zh,len_func = len)
def cut_string(input,max_len,len_func,split_func,tokenizer = False):
    if len_func(input)<= max_len:
        return [input]
    else:
        s = split_func(input)
        output = ['']
        for sen in s:
            if len_func(output[-1]+sen)>max_len:
                output.append('')
            else:
                output[-1] = output[-1]+sen
        return [x for x in output if check_quote_pair(x)]
def soft_cut_string(input,max_len,len_func,split_func,tokenizer = False):
    s = split_func(input)
    output = ['']
    for sen in s:
        if len_func(output[-1])>max_len:
            output.append('')
        else:
            output[-1] = output[-1]+sen
    return output

if __name__ == '__main__':
    print(check_quote_pair('杨杏园笑道：“差事倒是一个好差事，不过我那些朋友，因为我天天来，早造了许多谣言，如今索性教起书来，那不是给人家笑话吗？”梨云冷笑一声，说道：“我知道你不肯，不过白说一声。但是人家怎么天天去教书的呢？他就不怕给人家笑话吗！”杨杏园道：“人家教书有好处。我呢？”梨云脸一红，把鞋子轻轻的踢着杨杏园的脚，低低的笑着说道：“你又是瞎说。”他们正在这里软语缠绵，只听见花啦啦一阵响，好像打翻了许多东西。'))
