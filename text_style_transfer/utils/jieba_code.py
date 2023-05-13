import jieba
import jieba.posseg as pseg
import json
from tqdm import tqdm
import jieba.analyse
import re

def test():
    text = ["音蛙坐在田边突然看见一条肥牛，望着伟大的身躯，引起了它的嫉妒。", "它狂妄地鼓胀它的皮肤，尽可能地鼓得大而又大。", "肥牛劝阻不成，只能看着音蛙继续吹鼓自己的肚子。", "最终，音蛙鼓爆了自己的胸膛。"]
    # text = "".join(["山路狭窄，那骑马却横冲直撞，一下子将一个怀抱小孩的灾民妇人撞下路旁水中，马上乘者竟是毫不理会，自管策马疾驰而来。群雄俱各大怒。", "卫春华首先窜出，抢过去拉住骑者左脚一扯，将他拉下马来，劈面一拳，结结实实打在他面门之上。", "那人“哇”的一声，吐出一口血水、三只门牙。", "那人是个军官，站起身来，破口大骂：“你们这批土匪流氓，老子有紧急公事在身，回来再跟你们算帐。”上马欲行。", "章进在他右边一扯，又将他拉下马来，喝道：“甚么紧急公事，偏教你多等一会。”", "陈家洛道：“十哥，搜搜他身上，有甚么东西。”", "章进在他身上一抄，搜出一封公文。交了过去。", "陈家洛见是封插上鸡毛、烧焦了角的文书，知是急报公文，是命驿站连日连夜递送的，封皮上写着“六百里加急呈定边大将军兆”的字样，随手撕破火漆印，抽出公文。", "那军官见撕开公文，大惊失色，高叫起来：“这是军中密件，你不怕杀头吗？”"])
    # text = "有的无路可走，见大炮滚下来的声势险恶，踊身一跳，跌入了深谷。十尊大炮翻翻滚滚，向下直冲，越来越快。骡马在前疾驰，不久就被大炮赶上，压得血肉横飞。过了一阵，巨响震耳欲聋，十尊大炮都跌入深谷去了。雷蒙和彼得惊魂甫定，回顾若克琳时，见她已吓得晕了过去。两人救起了她，指挥士兵伏下抵敌。敌人早在坡上挖了深坑，用山泥筑成挡壁，火枪射去，伤不到一根毫毛，羽箭却不住嗖嗖射来。战了两个多时辰，洋兵始终不能突围。雷蒙道：“咱们火药不够用了，只得硬冲。”彼得道：“叫钱通四去问问，这些土匪到底要甚么。”雷蒙怒道：“跟土匪有甚么说的？你不敢去，我来冲。”彼得道：“土匪弓箭厉害，何必逞无谓的勇敢？”雷蒙望了若克琳一眼，恶狠狠的吐了口唾沫，骂道：“懦夫，懦夫！”彼得气得面色苍白，低声道：“等打退了土匪，叫你知道无礼的代价。”雷蒙一跃而起，叫道：“是好汉跟我来！”"
    # text = "".join(["很多年前，成群的藏羚羊生活在中国藏北高原上。", "它们和其他很多野生动物一样，是当地猎人的猎捕对象。", "一天清晨，老猎人巴布刚走出帐篷，就发现几步之遥的草坡上站立着一只肥壮的藏羚羊。", "真是送上门来的美事啊！他迅速地举起枪。", "奇怪的是，那只藏羚羊竟丝毫没有逃走的意思，只是用哀求的眼神望着巴布，然后冲着他前行两步，两条前腿扑通一声跪在地上。", "巴布心头一软，这只藏羚羊在求他饶命呢。", "可是一个优秀的猎人是不应该被猎物的可怜相打动的。", "猎人又仔细看了看发现藏羚羊肚子里有孩子，想到了自己母亲，就放过了藏羚羊。"])
    # text = "郭翰是古时候一名才子。一个夏日的晚上，他在院中乘凉。忽然，一阵风起，送来一股沁人心脾的清香，一位少女驾着白云从天而降，出现在郭翰眼前。如花似玉的美女，光彩夺目的纱衣，郭翰心中叫绝。少女说，她是织女，从天宫来。郭翰仔细打量着眼前的仙女，只见她身上的纱衣华丽而合身。奇怪的是，纱衣上连半点接缝都找不出来。仙女笑说，神仙的衣裳都不是用针线缝制的，自然没有接缝。郭翰觉得人间确实做不到这种手工艺，确信了仙女的身份。"
    # text = "".join(["滕一雷把袋里所剩的三枚制钱拿出来还给张召重，另外又取出一枚雍正通宝，顾哈两人拿出来的也都是雍正通宝。", "其时上距雍正不远，民间所用制钱，雍正通宝远较顺治通宝为多。", "陈家洛道：“我身边没带铜钱，就用张大哥这枚吧。”", "张召重道：“毕竟是陈当家的气度不同。四枚雍正通宝已经有了，顺治通宝就用这一枚。顾老二，你说成不成？”", "顾金标怒道：“不要顺治通宝！铜钱上顺治、雍正，字就不同，谁都摸得出来。”", "其实要在顷刻之间，凭手指抚摸而分辨钱上所铸小字，殊非易事，顾金标虽然明知，却终不免怀疑，又道：“你手里有一枚雍正通宝是白铜的，其余四枚都是黄铜的，谁拿到白铜的就是谁去。”", "张召重一楞，随即笑道：“一切依你！只怕还是轮到你去喂狼。”", "手指微一用力，已把白铜的铜钱捏得微有弯曲，和四枚黄铜的混在一起。", "顾金标怒道：“要是轮不到你我，咱俩还有一场架打！”", "张召重道：“当得奉陪。”"])
    # words = pseg.cut(text) #jieba默认模式
    # jieba.enable_paddle() #启动paddle模式。 0.40版之后开始支持，早期版本不支持
    # words = pseg.cut(text, use_paddle=True) #paddle模式
    # for word, flag in words:
    #     print('%s %s' % (word, flag))
    # a = jieba.analyse.extract_tags(text, topK=10, withWeight=False, allowPOS=("nz", 'ns', "nr", "nt", "nw", 'n', "PER", "LOC", "ORG"))
    # print(a)
    # b = jieba.analyse.textrank(text, topK=10, withWeight=False, allowPOS=("nz", 'ns', "nr", "nt", "nw", 'n', "PER", "LOC", "ORG"))
    # print(b)
    # for x, w in jieba.analyse.extract_tags(text, topK=5, withWeight=False, allowPOS=()):
    #     print('%s %s' % (x, w))
    sen_mask, mask_word = word_annotated(text)


def mask_ent(file, save):
    with open(file, 'r') as f:
        data = f.readlines()
    f_s = open(save, 'w')
    for line in tqdm(data):
        item = json.loads(line)
        text = item["text"]
        mask_text = []
        all_mask_words = []
        sen_mask, mask_word = word_annotated(text)
        # for sen in text:
        #     sen_mask, mask_word = word_annotated(sen)
        #     mask_text.append(sen_mask)
        #     all_mask_words = all_mask_words + mask_word
        item["text_mask"] = sen_mask
        item["mask_word"] = mask_word
        new_item = json.dumps(item, ensure_ascii=False)
        f_s.write(new_item + '\n')

    f_s.close()


def word_annotated(sentence):
    jieba.enable_paddle()  # 启动paddle模式。 0.40版之后开始支持，早期版本不支持
    # words = pseg.cut(sentence, use_paddle=True)  # paddle模式
    words = jieba.analyse.extract_tags("".join(sentence), topK=10, withWeight=False, allowPOS=("nz", 'ns', "nr", "nt", "nw", 'n', "PER", "LOC", "ORG"))  # paddle模式
    sens = " ".join(sentence)
    # tokens = jieba.cut(sens)
    # mask_keywords = []
    for word in words:
        sens = sens.replace(word, "<mask>")


    # for token in tokens:
    #     if token in words:
    #         sens = re.sub(token, "<mask>", sens)
    #         mask_keywords.append(token)
    # mask_sen = [sentence.replace()]
    # return sens.split(), mask_keywords
    return sens.split(), words

if __name__ == '__main__':
    file = "../data_ours/auxiliary_data/train.sen.add_index"
    save = "../data_ours/auxiliary_data/train.sen.add_index.mask"
    mask_ent(file, save)
    file = "../data_ours/auxiliary_data/test.sen.add_index"
    save = "../data_ours/auxiliary_data/test.sen.add_index.mask"
    # mask_label = ["nt", "nr", "nz", "nw", "PER"]
    mask_ent(file, save)
    # test()
