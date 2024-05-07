import pickle
import os
import random
import clip as clip

def subsetdata(filepath):
    if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
                random.seed(42)
                # reduced_filenames = random.sample(filenames, int(0.01 * len(filenames)))
                reduced_filenames = random.sample(filenames, int(0.1 * len(filenames)))
                print('Load filenames from: %s (%d)' % (filepath, len(reduced_filenames)))
            with open('filenames.pickle', 'wb') as f:
                pickle.dump(reduced_filenames, f)
    else:
            filenames = []
    return reduced_filenames

def get_caption(cap_path):
    eff_captions = []
    with open(cap_path, "r") as f:
        captions = f.read().encode('utf-8').decode('utf8').split('\n')
    for cap in captions:
        if len(cap) != 0:
            eff_captions.append(cap)
    # print(len(eff_captions))
    sent_ix = random.randint(0, len(eff_captions)-1)
    caption = eff_captions[sent_ix]
    # print(f"sent_ix: {sent_ix}")
    # print(f"eff_captions length: {len(eff_captions)}")
    # print(eff_captions)
    tokens = clip.tokenize(caption,truncate=True)
    # return caption, tokens[0]
    return caption

def main():
    filepath = 'filenames1.pickle'
    text_path = '../text'
    reduced_filenames = subsetdata(filepath)
    for idx in range(len(reduced_filenames)):
        key = reduced_filenames[idx]
        cap_path = '%s/%s.txt' % (text_path,key)
        print(get_caption(cap_path))        


if __name__=="__main__":
     main()