


def load_most_common_words(file, num):
    common_words = []       
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            word, fre = line.split()
            fre = int(fre)
            if fre >= num:
                common_words.append(word)
            else:
                break
    
    return common_words

def get_high_frequency_ch(files, nums, language):
    
    lx_words = load_most_common_words(files[0], nums[0]*0.1)
    jy_words = load_most_common_words(files[1], nums[1]*0.1)
    gs_words = load_most_common_words(files[2], nums[2]*0.1)
    style_unrelated_words = set(lx_words) & set(jy_words) & set(gs_words)
    with open("style_unrelated_words.{}".format(language), 'w') as f:
        for i in style_unrelated_words:
            f.write(i + "\n")
            
def get_high_frequency_en(files, nums, language):
    
    Shakespeare_words = load_most_common_words(files[0], nums[0]*0.1)
    roc_words = load_most_common_words(files[1], nums[1]*0.1)
    style_unrelated_words = set(Shakespeare_words) & set(roc_words)
    with open("style_unrelated_words.{}".format(language), 'w') as f:
        for i in style_unrelated_words:
            f.write(i + "\n")

if __name__ =="__main__":
    lx_num = 3036
    jy_num = 2964
    gs_num = 2427
    ch_files = ["chinese_words_frequency/word_frequency.lx", "chinese_words_frequency/word_frequency.jy", "chinese_words_frequency/word_frequency.gs"]
    ch_nums = [lx_num, jy_num, gs_num]
    # get_high_frequency_ch(ch_files, ch_nums, "ch")
    
    
    Shakespeare_num = 1161
    roc_num = 1741
    en_files = ["english_words_frequency/word_frequency.shakespeare", "english_words_frequency/word_frequency.story",]
    en_nums = [Shakespeare_num, roc_num]
    get_high_frequency_en(en_files, en_nums, "en")