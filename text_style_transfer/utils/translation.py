# -*- coding: UTF-8 -*-
import random
import re
import json
from tqdm import tqdm
# from googletrans import Translator
import time
from hashlib import md5
import requests
# import pandas as pd



LANGUAGES = {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'eu': 'basque',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    'co': 'corsican',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'ko': 'korean',
    'ku': 'kurdish (kurmanji)',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar (burmese)',
    'ne': 'nepali',
    'no': 'norwegian',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu',
    'fil': 'Filipino',
    'he': 'Hebrew'
}




def translation(file, save):
    # translator = Translator()
    # translator = Translator(service_urls=['translate.google.cn'])  # 如果可以上外网，还可添加 'translate.google.com' 等
    f_s = open(save, 'w')
    with open(file, 'r') as f:
        data = f.readlines()
        for i, line in enumerate(tqdm(data)):
            
            time.sleep(1)

            # item = json.loads(line)
            # text_list = item['text']
            # text = "\n".join(text_list)
            text = line.strip()
            result = Baidu_Text_trans(text)
            new_item = json.dumps(result, indent=4, ensure_ascii=False)
            # translations = translator.translate(text, src="zh-CN", dest='en')
            # cn2en = translations.text
            # print(cn2en)
            # eng_list = []
            # for text in text_list:
            #     time.sleep(1)
            #     cn2en = Baidu_Text_trans(text)
            #     eng_list.append(cn2en)
            # item['text'] = eng_list
            # new_item = json.dumps(item, ensure_ascii=False)
            f_s.write(new_item + '\n')
    f_s.close()

def Baidu_Text_trans(text):
    appid = '20210914000943453'
    appkey = 'DpkT6pZQMBNp4foPrRJq'

    # from_lang = 'zh'
    # to_lang = 'en'
    from_lang = 'en'
    to_lang = 'zh'

    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + text + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': text, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    # res = result['trans_result'][0]['dst']

    return result


def normal_cut_sentence(text):
    text = re.sub('([。！？\?])([^’”])',r'\1\n\2',text) #普通断句符号且后面没有引号
    text = re.sub('(\.{6})([^’”])',r'\1\n\2',text) #英文省略号且后面没有引号
    text = re.sub('(\…{2})([^’”])',r'\1\n\2',text) #中文省略号且后面没有引号
    text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])',r'\1\n\2',text) #断句号+引号且后面没有引号
    return text.split("\n")

def cut_sentence_with_quotation_marks(text):
    p = re.compile("“.*?”")
    list = []
    index = 0
    length = len(text)
    for i in p.finditer(text):
        temp = ''
        start = i.start()
        end = i.end()
        if start == 0 and end == length:
            break


        temp = text[index:start]
        if temp != '':
            temp_list = normal_cut_sentence(temp)
            if len(list) != 0 and list[-1][-2] not in ["。", "！", "？", "?", "…", "."]:
                m = temp_list.pop(0)
                n = list.pop(-1)
                new_n_m = n + m
                list.append(new_n_m)
            list += temp_list

        temp = text[start:end]
        if temp != ' ':
            if len(list) != 0 and list[-1][-1] not in ["。", "！", "？", "?", "……", "......"]:
                temp = list[-1] + temp
                list.pop()
            list.append(temp)
        index = end

    if index+1 < length:
        temp = text[index:]
        temp_list = normal_cut_sentence(temp)
        if len(list) != 0 and list[-1][-2] not in ["。", "！", "？", "?", "……", "......"]:
            m = temp_list.pop(0)
            n = list.pop(-1)
            new_n_m = n + m
            list.append(new_n_m)
        list += temp_list

    re_list = []
    for i, sen in enumerate(list):
        if i == 0:
            re_list.append(sen)
            continue
        if len(sen) < 10:
            m = re_list.pop()
            m_n = m + sen
            re_list.append(m_n)
        else:
            re_list.append(sen)


    return re_list


def cut_dataset2sentence(file, save):
    f_s = open(save, 'w')
    with open(file, "r") as f:
        data = f.readlines()
        for line in tqdm(data):
            item = json.loads(line)
            text = item["text"]
            sen_list = cut_sentence_with_quotation_marks(text)
            item["text"] = sen_list
            new_item = json.dumps(item, ensure_ascii=False)
            f_s.write(new_item + '\n')

    f_s.close()



def concat_files(files, save):
    data_list = []
    for file in files:
        with open(file, 'r') as f:
            data = f.readlines()
            data_list += data
    with open(save, 'w') as f:
        for line in data_list:
            f.write(line)

def statistics_sen(file):
    n = 0
    with open(file, 'r') as f:
        data =f.readlines()
        for line in data:
            item = json.loads(line)
            text = item["text"]
            if len(text) < 2:
                print(text)
                n += 1
    print(n)



def load_json_multiple(segments):
    chunk = ""
    for segment in segments:
        chunk += segment
        try:
            yield json.loads(chunk)
            chunk = ""
        except ValueError:
            pass


def filter2eng(file, save):
    f_s = open(save, 'w')
    with open(file, 'r') as f:
        for parsed_json in load_json_multiple(f):
            try:
                trans_res_list = parsed_json["trans_result"]
            except:
                print(parsed_json)
                continue
            cn2eng_list = []
            for res in trans_res_list:
                cn2eng = res["dst"]
                cn2eng_list.append(cn2eng)
            new_item = {"text": cn2eng_list}
            new_item = json.dumps(new_item)
            f_s.write(new_item + "\n")
    f_s.close()

def badcase_translation(file, save):
    all_data = []
    with open(file, 'r') as f:
        data = f.readlines()
        for line in data:
            # item = json.loads(line)
            all_data.append(line)


    bad_case = ["郭芙道：“妹妹给杨过抱了去啦，他还抢了我的小红马去。你瞧这把剑。”", "说着举起手中弯剑，道：“他用断臂的袖子一拂，这剑撞在墙角上，便成了这个样子。”", "黄蓉与李莫愁齐声道：“是袖子？”", "郭芙道：“是啊，当真邪门！想不到他又学会了妖法。”", "黄蓉与李莫愁相视一眼，均各骇然。", "她二人自然都知一人内力练到了极深湛之境，确可挥绸成棍、以柔击刚，但纵遇明师，天资颖异，至少也得三四十年的功力，杨过小小年纪，竟能到此境地，实是罕有。", "黄蓉听说女儿果然是杨过抱了去，倒放了一大半心。", "李莫愁却自寻思：“这小子功夫练到这步田地，定是得力于我师父的玉女心经。眼下有郭夫人这个强援，我助她夺回女儿，她便得助我夺取心经。我是本派大弟子，师妹虽得师父喜爱，但她连犯本派门规，这心经焉能落入男子手中？”", "她这么一想，自己颇觉理直气壮。"]
    res_list = []
    text = "\n".join(bad_case)
    result = Baidu_Text_trans(text)
    cn2eng_list = result["trans_result"]
    for res in cn2eng_list:
        eng = res["dst"]
        res_list.append(eng)

    insert = {"text": res_list}
    insert = json.dumps(insert) + '\n'
    all_data.insert(5777, insert)
    with open(save, 'w') as f:
        for line in all_data:
            f.write(line)




if __name__ == '__main__':

    # text = "一灯大师瞧了杨过一眼，也十分诧异。慈恩厉声喝道：“你是谁？干甚么？”杨过道：“尊师好言相劝，大师何以执迷不悟？不听金玉良言，已是不该，反而以怨报德，竟向尊师下毒手，如此为人，岂非禽兽不如？”慈恩大怒，喝道：“你也是丐帮的？跟那个鬼鬼祟祟的长老是一路的么？”杨过笑道：“这二人是丐帮败类，大师除恶即是行善，何必自悔？”慈恩一怔，自言自语：“除恶即是行善。。。。。。除恶即是行善。。。。。。”杨过隔着板壁听他师徒二人对答，已隐约明白了他的心事，知他因悔生恨，恶念横起，又道：“那二人是丐帮叛徒，意引狼入室，将我大汉河山出荬于异族。大师杀此二人，实是莫大功德。这二人不死，不知有多少善男信女家破人亡。我佛虽然慈悲，但遇到邪魔外道，不也要大显神通将之驱灭么？”杨过所知的佛学尽此而已，实是浅薄之至，但慈恩听来却极为入耳。"
    # text_1 = "但已经使又一部分人很不高兴了，就招来了两标军马的围剿。创造社竖起了“为艺术的艺术”的大旗，喊着“自我表现”的口号，要用波斯诗人的酒杯，“黄书”文士的手杖，将这些“庸俗”打平。还有一标是那些受过了英国的小说在供绅士淑女的欣赏，美国的小说家在迎合读者的心思这些“文艺理论”的洗礼而回来的，一听到下层社会的叫唤和呻吟，就使他们眉头百结，扬起了带着白手套的纤手，挥斥道：这些下流都从“艺术之宫”里滚出去！"
    # text_2 = "英文Assassin（注：）（刺客、暗杀者）一字就由此而来。旭烈兀攻破了该派在高峰上的城堡，一举而将之歼灭，不分老小，全部杀光。但这教派分布甚广，总部被摧毁后仍在别的地方继续恐怖活动。那时回教徒在中东一带势力极大。回教的大教主称为哈里发，驻在巴格达（今伊拉克首都），就像基督教的教皇驻在罗马一样。哈里发统率大军，兼管政治。当时在巴格达统治已近五百年，又占领了基督教的圣城耶路撒冷。西欧的基督徒组织“十字军东征”，一次又一次的和回教徒作战，规模巨大的东征共有八次，但终于打不过回教徒而失败。旭烈兀的西征却只打一仗就摧毁了回教的大本营。(21)那个哈里发名叫木司塔辛，爱好音乐，是大食朝的第三十七代哈里发。一说旭烈兀将他裹在毛毡中，放在巴格达大街上，命军士纵马践踏而死。"
    # text_3 = "森林中有一条小河，小猴子、小狗熊一跃就过去了。可小白兔、小老鼠太小了，跳不过河，常常发愁。大象看见了，就用长长的鼻子把它们卷起来，送到河对岸。小白兔、小老鼠很高兴地说大象鼻子坐着特别舒服。小猴子、小狗熊也很羡慕，假装生病让大象送。但是小狗熊太重了，大象因此累坏了。于是，小猴子想到了一个办法。他们一起建了一座象鼻桥，给大家来过河。从此再也不用麻烦大象了。"
    # text_4 = "岳不群站在台角，只是微笑。人人都看了出来，左冷禅确是双眼给岳不群刺瞎了，自是尽皆惊异无比。只有令狐冲和盈盈，才对如此结局不感诧异。岳不群长剑脱手，此后所使的招术，便和东方不败的武功大同小异。那日在黑木崖上，任我行、令狐冲、向问天、上官云四人联手和东方不败相斗，尚且不敌，直到盈盈转而攻击杨莲亭，这才侥幸得手，饶是如此，任我行终究还是被刺瞎了一只眼睛，当时生死所差，只是一线。岳不群身形之飘忽迅捷，比之东方不败虽然颇有不如，但料到单打独斗，左冷禅非输不可，果然过不多时，他双目便被针刺瞎。令狐冲见师父得胜，心下并不喜悦，反而突然感到说不出的害怕。岳不群性子温和，待他向来亲切，他自小对师父挚爱实胜于敬畏。后来师父将他逐出门墙，他也深知自己行事乖张任性，实是罪有应得，只盼能得师父师娘宽恕，从未生过半分怨怼之意。"
    # text_5 = "“对。太太。我也这样想。明天我想起得早些。倘若你醒得早，那就叫醒我。我准备再远走五十里，看看可有些獐子兔子。……但是，怕也难。当我射封豕长蛇的时候，野兽是那么多。你还该记得罢，丈母的门前就常有黑熊走过，叫我去射了好几回……。”"
    # text_6 = "“女吊”也许是方言，翻成普通的白话，只好说是“女性的吊死鬼”。其实，在平时，说起“吊死鬼”，就已经含有“女性的”的意思的，因为投缳而死者，向来以妇人女子为最多。有一种蜘蛛，用一枝丝挂下自己的身体，悬在空中，《尔雅》上已谓之“蚬，缢女”，可见在周朝或汉朝，自经的已经大抵是女性了，所以那时不称它为男性的“缢夫”或中性的“缢者”。不过一到做“大戏”或“目连戏”的时候，我们便能在看客的嘴里听到“女吊”的称呼。也叫作“吊神”。横死的鬼魂而得到“神”的尊号的，我还没有发见过第二位，则其受民众之爱戴也可想。但为什么这时独要称她“女吊”呢？很容易解：因为在戏台上，也要有“男吊”出现了。"
    # text_7 = "木高峰脸上现出诧异神情，道：“甚么？凭这小子这一点儿微末道行，居然能去救灵珊侄女？只怕这话要倒过来说，是灵珊贤侄女慧眼识玉郎……”岳不群知道这驼子粗俗下流，接下去定然没有好话，便截住他话头，说道：“江湖上同道有难，谁都该当出手相援，粉身碎骨是救，一言相劝也是救，倒也不在乎武艺的高低。木兄，你如决意收他为徒，不妨让这少年禀明了父母，再来投入贵派门下，岂不两全其美？”"

    # translator = Translator()
    # translations = translator.translate(text, dest='en')
    # print(translations.text)
    # en2cn = translator.translate(translations.text, dest='zh-cn').text
    # print(en2cn)
    # print('----------')
    # print(text)

    # file = "data_ours/final_data/train.json"
    # save = "data_ours/auxiliary_data/train.en"
    # save_sen = "data_ours/auxiliary_data/train.sen"
    # file = "data_ours/final_data/test.json"
    # save_sen = "data_ours/auxiliary_data/test.sen"
    file = "data_ours/final_data/valid.json"
    save_sen = "data_ours/auxiliary_data/valid.sen"
    cut_dataset2sentence(file, save_sen)

    # cat two files
    # files = ["data_ours/auxiliary_data/train.en_4786", "data_ours/auxiliary_data/train.en"]
    # save = "data_ours/auxiliary_data/train.eng"
    # concat_files(files, save)

    
    # file = "data_ours/auxiliary_data/train.sen"
    # save = "data_ours/auxiliary_data/train.sen.temp"
    # translation(file, save)

    # file = "data_ours/auxiliary_data/train.sen.cn2eng"
    # save = "data_ours/auxiliary_data/train.sen.eng"
    # filter2eng(file, save)


    # file = "data_ours/auxiliary_data/train.sen.v1"
    # statistics_sen(file)

    # file = "data_ours/auxiliary_data/train.sen.eng"
    # badcase_translation(file, file)

    # file = "data_ours/auxiliary_data/train.sen.eng"
    # save = "data_ours/auxiliary_data/train.sen.eng2cn"


    # file = "data_ours/auxiliary_data/train.eng"
    # save = "data_ours/auxiliary_data/train.eng2cn"
    #
    #
    #
    # translation(file, save)


