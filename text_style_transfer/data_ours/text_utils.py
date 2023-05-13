import glob,os,sys
LQZEICH='LQCYM'
RQZEICH='RQCYM'
SPLITZEICH='SCHWEIGSSS'
SPLITMARK='？ ? . 。 , ！ ! …… ，'.split(' ')
ENDMARK='？ ? 。 ！ ! ……'.split(' ')
LEGALMARK= '？ ? 。 ！ ! …… ， , ; ； 、'.split(' ')
SENLEN = 5
def is_quote_complete(string):
    return string.count('“')==string.count('”')
def if_notrepeat(string):
    slotlen = min(20,int(len(string)/3.0))
    if string.find(string[:slotlen],slotlen)!= -1:
        return False
    else:
        return True
def is_improper_marks(string):
    if string.find('。：') !=-1:
        return True
    if string.find('。。。。。。') !=-1:
        return True
    return False
def is_annotation(string):
    import re
    if len(re.findall('\[.\]',string))>0:
        return True
    return False
def is_chaptername(string):
    import re
    if len(re.findall('第.章',string))>0:
        return True
    return False
def is_chinese(char):
    if '\u4e00' <= char <= '\u9fff':
        return True
    return False
def is_multend(string):
    import re
    if len(re.findall('['+''.join(LEGALMARK)+']{3}',string))>0:
        return True

    return False
def chinese_ratio(string):
    return len([1 for x in string if is_chinese(x)])/float(len(string))
def is_firstpronoun(string):
    return  len([x for x in string if x == '我'])/len(string) >0.004    
def clean_spam(string):
    s = split_by(string,[' ','。','.','\xa0','_','xx','|','/'])
    return ''.join([x for x in s if len(x)>3 and chinese_ratio(x)>0.5])
def isnot_story(string):
    if string.find('读后感')!=-1:
        return True
    if string.find('读书笔记')!=-1:
        return True
    return False
def paragraph_sanity_check(string):
    return is_quote_complete(string) \
    and if_notrepeat(string) \
    and chinese_ratio(string)>0.5 \
    and not is_firstpronoun(string)\
    and not is_chaptername(string)
    '''and not is_annotation(string)
    and not is_improper_marks(string)
    and not isnot_story(string)'''
def modify_illegal_sentence(string):
    # import pdb;pdb.set_trace()
    s = split_by_fullstop(string)
    ssc =  [sentence_sanity_check(x) for x in s]
    index = list(range(len(s)))
    index = [str(x) if ssc[x] else 'k' for x in index ]
    index = '/'.join(index).split('k')
    index = max(index, key=len)
    s = [s[i] for i in [int(x) for x in index.split('/') if len(x)>0] ]
    return ''.join(s)

def del_repeats_in_list(s):
    def if_repeat(string,s):
        for a in split_by_fullstop(string):
            for b in s:
                if a in b:
                    return True
        return False
    index = []
    for i,a in enumerate(s):
        index.append(if_repeat(a,s[i+1:]))
    return [s[i] for i in range(len(s)) if not index[i] ]
                






def sentence_sanity_check(string):
    for x in string:
        if not is_chinese(x):
            if x not in LEGALMARK:
                return False
    
    return not is_multend(string)
def nested_dict():
    import collections
    return collections.defaultdict(nested_dict)
def complete_quote(string):
    if string.startswith('“') and not string.endswith('”'):
        return string + '”'
    elif not string.startswith('“') and string.endswith('”'):
        return '“'+string
    else:
        return string
def is_pureword(string):
    for mark in SPLITMARK:
        if mark in string:
            return False
    return True


def is_dialogquote(string):
    if string.startswith('“') and string.endswith('”') and not is_pureword(string):
        return True
    else:
        return False
def is_normalquote(string):
    if string.startswith('“') and string.endswith('”') and is_pureword(string):
        return True
    else:
        return False
def is_sentence_complete(string):
    for mark in ENDMARK:
        if string.endswith(mark):
            return True
    return False
def split_by(string,marks):
    s = string
    for mark in marks:

        s=s.replace(mark,mark+SPLITZEICH)
    return [x for x in s.split(SPLITZEICH) if len(x)>0]
def split_by_fullstop(string):
    
    return split_by(string,ENDMARK)
def split_by_pair(string,left,right):
    s=string.replace(left,SPLITZEICH+left)
    s=s.replace(right,right+SPLITZEICH)
    s = s.split(SPLITZEICH)
    return s
def split_by_quote(string):
    s=string.replace('“',SPLITZEICH+'“')
    s=s.replace('”','”'+SPLITZEICH)
    s = s.split(SPLITZEICH)
    return s
def replace_quote(string):
    s = string.replace('“',LQZEICH)
    s = s.replace('”',RQZEICH)
    return s
def undoreplace_quote(string):
    s = string.replace(LQZEICH,'“')
    s = s.replace(RQZEICH,'”')
    return s
def split_by_dialogquote(string):
    s = split_by_quote(string)
    
    s = [replace_quote(x) if is_normalquote(x) else x for x in s]
    s= ''.join(s)
    s = split_by_quote(s)
    s = [undoreplace_quote(x) for x in s]
    return s

def strip_incomplete_sentence(string):
    s = string
    

    
    s = split_by_fullstop(string)
    st = 0
    # if s[0].find('说')!=-1 or len(s[0])<5:
    #     st = 1
    if not is_sentence_complete(s[-1]):
        s =''.join(s[st:-1])

    else:
        s =''.join(s[st:])
    
    while len(s)>1:
        # import pdb;pdb.set_trace()

        if (not is_chinese(s[0])) :
            s = s[1:]
        else:
            return s
    if len(s)==1 and not is_chinese(s):
        return ''
    return s
     
def count_clauses(string):
    return len(split_by_fullstop(string))   
def count_sentences(string):
    return len(split_by(string,ENDMARK)) 
def get_word_len(s):
    import jieba
    return len(list(jieba.cut(s, cut_all=False)))
