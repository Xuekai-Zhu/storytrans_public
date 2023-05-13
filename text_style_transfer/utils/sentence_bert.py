import json
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm


def sentence_emb(file, save):
    model = SentenceTransformer('all-MiniLM-L12-v2')
    all_data = []
    with open(file, "r") as f:
        data = f.readlines()
        for line in tqdm(data):
            item = json.loads(line)
            text_list = item["text"]
            sentence_embeddings = model.encode(text_list)
            all_data.append(sentence_embeddings)

    with open(save, 'wb') as f:
        pickle.dump(all_data, f)



if __name__ == '__main__':
    file = "data_ours/auxiliary_data/train.sen.eng"
    save = "data_ours/auxiliary_data/train.sen.emb.pickle"
    sentence_emb(file, save)